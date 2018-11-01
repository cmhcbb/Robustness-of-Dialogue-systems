# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Dialogue runner class. Implementes communication between two Agents.
"""
import sys
import pdb
import logging
import numpy as np

from metric import MetricsContainer
import data
import utils
import domain
from torch.autograd import Variable
import torch, random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(filename)s : %(message)s', level=logging.INFO)

class DialogLogger(object):
    """Logger for a dialogue."""
    CODE2ITEM = [
        ('item0', 'book'),
        ('item1', 'hat'),
        ('item2', 'ball'),
    ]

    def __init__(self, verbose=False, log_file=None, append=False):
        self.logs = []
        if verbose:
            self.logs.append(sys.stderr)
        if log_file:
            flags = 'a' if append else 'w'
            self.logs.append(open(log_file, flags))

    def _dump(self, s, forced=False):
        for log in self.logs:
            print(s, file=log)
            log.flush()
        if forced:
            print(s, file=sys.stdout)
            sys.stdout.flush()

    def _dump_with_name(self, name, s):
        self._dump('{0: <5} : {1}'.format(name, s))

    def dump_ctx(self, name, ctx):
        assert len(ctx) == 6, 'we expect 3 objects'
        s = ' '.join(['%s=(count:%s value:%s)' % (self.CODE2ITEM[i][1], ctx[2 * i], ctx[2 * i + 1]) \
            for i in range(3)])
        self._dump_with_name(name, s)

    def dump_sent(self, name, sent):
        self._dump_with_name(name, ' '.join(sent))

    def dump_choice(self, name, choice):
        def rep(w):
            p = w.split('=')
            if len(p) == 2:
                for k, v in self.CODE2ITEM:
                    if p[0] == k:
                        return '%s=%s' % (v, p[1])
            return w

        self._dump_with_name(name, ' '.join([rep(c) for c in choice]))

    def dump_agreement(self, agree):
        self._dump('Agreement!' if agree else 'Disagreement?!')

    def dump_reward(self, name, agree, reward):
        if agree:
            self._dump_with_name(name, '%d points' % reward)
        else:
            self._dump_with_name(name, '0 (potential %d)' % reward)

    def dump(self, s, forced=False):
        self._dump(s, forced=forced)


class DialogSelfTrainLogger(DialogLogger):
    """This logger is used to produce new training data from selfplaying."""
    def __init__(self, verbose=False, log_file=None):
        super(DialogSelfTrainLogger, self).__init__(verbose, log_file)
        self.name2example = {}
        self.name2choice = {}

    def _dump_with_name(self, name, sent):
        for n in self.name2example:
            if n == name:
                self.name2example[n] += " YOU: "
            else:
                self.name2example[n] += " THEM: "

            self.name2example[n] += sent

    def dump_ctx(self, name, ctx):
        self.name2example[name] = ' '.join(ctx)

    def dump_choice(self, name, choice):
        self.name2choice[name] = ' '.join(choice)

    def dump_agreement(self, agree):
        if agree:
            for name in self.name2example:
                for other_name in self.name2example:
                    if name != other_name:
                        self.name2example[name] += ' ' + self.name2choice[name]
                        self.name2example[name] += ' ' + self.name2choice[other_name]
                        self._dump(self.name2example[name])

    def dump_reward(self, name, agree, reward):
        pass


class Dialog(object):
    """Dialogue runner."""
    def __init__(self, agents, args):
        # for now we only suppport dialog of 2 agents
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()

    def _register_metrics(self):
        """Registers valuable metrics."""
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_percentage('agree')
        self.metrics.register_average('advantage')
        self.metrics.register_time('time')
        self.metrics.register_average('comb_rew')
        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_percentage('%s_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        ref_text = ' '.join(data.read_lines(self.args.ref_text))
        self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 1 and out[0] == '<selection>'

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def run(self, ctxs, logger):
        """Runs one instance of the dialogue."""
        assert len(self.agents) == len(ctxs)
        # initialize agents by feeding in the contexes
        #for agent, ctx in zip(self.agents, ctxs):
        #    agent.feed_context(ctx)
        #   logger.dump_ctx(agent.name, ctx)
        self.agents[0].feed_context(ctxs[0])
        logger.dump_ctx(self.agents[0].name, ctxs[0])
        self.agents[1].feed_context(ctxs[1],ctxs[0])
        logger.dump_ctx(self.agents[1].name, ctxs[1])

        logger.dump('-' * 80)

        # choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        writer, reader = self.agents

        conv = []
        # reset metrics
        self.metrics.reset()

         #### Minhao ####
        count_turns = 0       

        while True:
            # produce an utterance
            if count_turns > self.args.max_turns-1:
                if writer == self.agents[0]:
                    inpt_emb, inpt, lang_hs, lang_h, words = writer.write_white(reader)
                    #print(writer.words[-1][0].grad)
                    ### need padding in the input_emb
                    break
                #else:

            else:
                out = writer.write()

            self.metrics.record('sent_len', len(out))
            self.metrics.record('full_match', out)
            self.metrics.record('%s_unique' % writer.name, out)

            # append the utterance to the conversation
            conv.append(out)
            # make the other agent to read it
            reader.read(out)
            if not writer.human:
                logger.dump_sent(writer.name, out)
            # check if the end of the conversation was generated
            if self._is_selection(out):
                self.metrics.record('%s_sel' % writer.name, 1)
                self.metrics.record('%s_sel' % reader.name, 0)
                break
            writer, reader = reader, writer
            count_turns += 1
            ##### add selection mark if exceeding the max_turns

        ### Minhao: need to design loss focusing on the choices
        ### No evalution in the conversation????

        bob = self.agents[1]

        choices = []
        #class_losses = []
        # generate choices for each of the agents
        #flag_loss=False
        
        ####
        # get the final loss and do the back-propagation
        c=1
        step_size = 5e-2   
        #lang_hs[0].retain_grad()
        #print(words)
        all_index_n = len(self.agents[0].model.word_dict)
        all_index = torch.LongTensor(range(all_index_n)).cuda()
        all_word_emb = self.agents[0].model.word_encoder(all_index)
        threshold = 2.5
        #print(all_word_emb.size())
        print(inpt_emb.size(),inpt)

        def get_embedding(inpt):
            prefix = Variable(torch.LongTensor(1).unsqueeze(1))
            prefix.data.fill_(bob.model.word_dict.get_idx("THEM:"))
            inpt = torch.cat([bob.model.to_device(prefix), inpt])
            inpt_emb = bob.model.word_encoder(inpt)
            return inpt_emb
        changed = False
        iterations = 200
        mask= [0] * (inpt_emb.size()[0]-1)
        for iter_idx in range(iterations):
            #print(inpt,len(bob.lang_hs),bob.lang_h.size())
            #print(len(bob.lang_hs))
            if changed:
                print(inpt)
                inpt_emb = bob.model.get_embedding(inpt,bob.lang_h,bob.ctx_h)
                changed = False
            inpt_emb.retain_grad()
            #bob.lang_hs[-1].retain_grad()
            bob.read_emb(inpt_emb, inpt)
            #print(len(bob.lang_hs))
            loss1, bob_out, _ = bob.write_selection(wb_attack=True)
            #print(len(bob.lang_hs))
            #print(len(lang_hs))
            bob.words = words.copy()
            #print(len(lang_hs))
            bob_choice, classify_loss, _ = bob.choose(inpt_emb=inpt_emb,wb_attack=True)
            #print(len(bob.lang_hs))
            t_loss = c*loss1 + classify_loss
            if (iter_idx+1)%10==0 and loss1==0.0 and classify_loss<=0.0:
                print("get legimate adversarial example")
                print(bob._decode(inpt,bob.model.word_dict))
                break
            #t_loss = loss1
            #bob.lang_hs[2].retain_grad()
            #logits.retain_grad()
            #t_loss = classify_loss
            t_loss.backward(retain_graph=True)
            #print(len(bob.lang_hs))
            
            #print(logits.grad)
            print(t_loss.item(),loss1.item(),classify_loss.item())
        #print(inpt_emb.size())
            #print(inpt_emb.grad.size())
            inpt_emb.grad[:,:,256:] = 0
            inpt_emb.grad[0,:,:] = 0
            #print(inpt_emb.grad[2])
            #inpt_emb.grad[0][:][:]=0
            inpt_emb = inpt_emb - step_size * inpt_emb.grad

            # projection
            if iter_idx%10==0:
                for emb_idx in range(1,inpt_emb.size()[0]):
                    rep_candidate = []
                    dis_a=[] 
                    for r_idx in range(all_index_n):
                        if r_idx==inpt[emb_idx-1].item():
                            continue
                        dis=torch.norm(inpt_emb[emb_idx][:,:256]-all_word_emb[r_idx]).item()
                        if dis< threshold:
                            rep_candidate.append(r_idx)
                            if not dis_a:
                                continue
                            elif dis<min(dis_a):
                                min_idx = r_idx

                        dis_a.append(dis)
                    #print(np.argmin(dis_a),min(dis_a))
                    if rep_candidate and mask[emb_idx-1]==0:
                        #mask[emb_idx-1]=1
                        #temp= random.choice(rep_candidate)
                        temp = min_idx
                        inpt[emb_idx-1]=temp
                        changed = True
                    #break
            #print(rep_candidate)

            bob.lang_hs = lang_hs.copy()
            bob.lang_h = lang_h.clone()
            
        #print(t_loss,lang_hs[0].grad)
        print("attack finished")
        #####
        choices.append(bob_choice)
        alice_choice = bob_choice[:]
        for indx in range(3):
           alice_choice[indx+3], alice_choice[indx] = alice_choice[indx], alice_choice[indx+3]
        choices.append(alice_choice) ######## always agree
        choices[1], choices[0] = choices[0], choices[1]
        #print(choices)
        #for agent in self.agents:
        #    choice, class_loss = agent.choose(flag=flag_loss)
        #    class_losses.append(class_loss)
        #    choices.append(choice)
        #    logger.dump_choice(agent.name, choice[: self.domain.selection_length() // 2])
        #    flag_loss=True

        # evaluate the choices, produce agreement and a reward
        #print(choices,ctxs)
        agree, rewards = self.domain.score_choices(choices, ctxs)
        logger.dump('-' * 80)
        logger.dump_agreement(agree)
        #print(rewards)
        # perform update, in case if any of the agents is learnable
        # let the difference become new reward
        ## how to combine the loss to the reward

        '''
        diff = rewards[0]-rewards[1] 
        flag = True
        agree = 1
        #print(5 - classify_loss.item())
        for agent, reward in zip(self.agents, rewards):           
            if flag:
                logger.dump_reward(agent.name, agree, reward)
                #agent.update(agree, 50-class_losses[1].item())
                agent.update(agree, 5-classify_loss.item())
                #agent.update(agree, diff - 0.05 * class_losses[1].data[0])
                #agent.update(agree, diff)
            else:
                logger.dump_reward(agent.name, agree, reward)
                if not self.args.fixed_bob:
                    agent.update(agree, reward)
            flag=False
        '''
        agree = 1
        for agent, reward in zip(self.agents, rewards):
            logger.dump_reward(agent.name, agree, reward)
            logging.debug("%s : %s : %s" % (str(agent.name), str(agree), str(rewards)))
            agent.update(agree, 5-classify_loss.item())


        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])
        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)
        for ctx, choice in zip(ctxs, choices):
            logger.dump('debug: %s %s' % (' '.join(ctx), ' '.join(choice)))

        return conv, agree, rewards

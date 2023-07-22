import numpy as np

from egttools.games import AbstractTwoPLayerGame


class CentipedeGame(AbstractTwoPLayerGame):
    
    def __init__(self, 
                 payoffs_pl1 : np.ndarray,
                 payoffs_pl2 : np.ndarray,
                 strategies : np.ndarray,
                ):
        self.payoffs_pl1 = payoffs_pl1
        self.payoffs_pl2 = payoffs_pl2
        assert len(payoffs_pl1) == len(payoffs_pl2)
        
        self.nb_steps = len(payoffs_pl1) - 1
        self.nb_actions = self.nb_steps//2 + 1
        
        self.strategies = strategies
        self.strategies_ = strategies
        self.nb_strategies_ = len(strategies)
            
        AbstractTwoPLayerGame.__init__(self, self.nb_strategies_)
    
    def get_min_take(self,
                     take_pl1 : np.ndarray,
                     take_pl2 : np.ndarray,
                    ):
        assert len(take_pl1)==len(take_pl2)
        assert len(take_pl1)==self.nb_steps+1
        
        min_take_cdf = np.zeros(self.nb_steps+1, dtype = float)
        
        min_take_cdf[0] = take_pl1[0]
        for i in range(1, self.nb_steps+1):
            min_take_cdf[i] = 1. - take_pl1[i+1:].sum() * take_pl2[i+1:].sum()
        min_take_cdf[-1] = 1.
        
        min_take_pdf = np.ediff1d(min_take_cdf, to_begin = min_take_cdf[0])
        assert np.isclose(min_take_pdf.sum(), 1.)
       
        return min_take_pdf
    
    
    def zero_padding_pl1(self, p : np.ndarray):
        # [p_0, 0, p_2, ..., 0, p_n]
        return np.insert(p, slice(1, None), 0.)
    
    def zero_padding_pl2(self, p : np.ndarray):
        # [0, p_1, ..., p_{n-1}, p_n]
        tmp = np.insert(p[:-1], slice(1, None), 0.)
        tmp = np.append(tmp, p[-1])
        return np.insert(tmp, 0, 0.)
    
    
    def get_take_distributions(self) -> np.ndarray:
        take_distribution_matrix = np.zeros((self.nb_strategies_, self.nb_strategies_, self.nb_steps+1), dtype = float)
        for i, strategy_a in enumerate(self.strategies):
            assert np.isclose(strategy_a[:self.nb_actions].sum(), 1.)
            assert np.isclose(strategy_a[self.nb_actions:].sum(), 1.)

            p_a_as_pl1 = self.zero_padding_pl1(strategy_a[:self.nb_actions])
            p_a_as_pl2 = self.zero_padding_pl2(strategy_a[self.nb_actions:])
            
            for j, strategy_b in enumerate(self.strategies):
                assert np.isclose(strategy_b[:self.nb_actions].sum(), 1.)
                assert np.isclose(strategy_b[self.nb_actions:].sum(), 1.)

                # A = pl.1, B = pl.2
                p_b_as_pl2 = self.zero_padding_pl2(strategy_b[self.nb_actions:])
                
                # A = pl.2, B = pl.1
                p_b_as_pl1 = self.zero_padding_pl1(strategy_b[:self.nb_actions])
                
                take_distribution_matrix[i,j]=self.get_min_take(p_a_as_pl1,p_b_as_pl2)+self.get_min_take(p_b_as_pl1,p_a_as_pl2)
     
        return .5 * take_distribution_matrix
    
    
    def get_unconditional_take_distributions(self) -> np.ndarray:
        take_distribution_matrix = np.zeros((self.nb_strategies_, self.nb_strategies_, self.nb_steps+1), dtype = float)
        for i, strategy_a in enumerate(self.strategies):
            assert np.isclose(strategy_a[:self.nb_actions].sum(), 1.)
            assert np.isclose(strategy_a[self.nb_actions:].sum(), 1.)

            p_a_as_pl1 = self.zero_padding_pl1(strategy_a[:self.nb_actions])
            p_a_as_pl2 = self.zero_padding_pl2(strategy_a[self.nb_actions:])
            
            for j, strategy_b in enumerate(self.strategies):
                assert np.isclose(strategy_b[:self.nb_actions].sum(), 1.)
                assert np.isclose(strategy_b[self.nb_actions:].sum(), 1.)

                # A = pl.1, B = pl.2
                p_b_as_pl2 = self.zero_padding_pl2(strategy_b[self.nb_actions:])
                
                # A = pl.2, B = pl.1
                p_b_as_pl1 = self.zero_padding_pl1(strategy_b[:self.nb_actions])
                
                take_distribution_matrix[i,j]= self.get_min_take(.5 * (p_a_as_pl1 + p_a_as_pl2), .5 * (p_b_as_pl1 + p_b_as_pl2))
     
        return take_distribution_matrix
    
    
    def calculate_payoffs_pl(self,
                            player : int, # 1 or 2
                            ) -> np.ndarray:
        
        pay_pl = np.zeros((self.nb_strategies_, self.nb_strategies_), dtype = float)
        
        for i, strategy_a in enumerate(self.strategies):
            
            if player == 1:
                p_a_as_pl1 = strategy_a[:self.nb_actions]
                assert np.isclose(p_a_as_pl1.sum(), 1.)
                p_a_as_pl1 = self.zero_padding_pl1(p_a_as_pl1)
            elif player == 2:
                p_a_as_pl2 = strategy_a[self.nb_actions:]
                assert np.isclose(p_a_as_pl2.sum(), 1.)
                p_a_as_pl2 = self.zero_padding_pl2(p_a_as_pl2)
            
            
            for j, strategy_b in enumerate(self.strategies):
                if player == 1:
                    p_b_as_pl2 = strategy_b[self.nb_actions:]
                    assert np.isclose(p_b_as_pl2.sum(), 1.)
                    p_b_as_pl2 = self.zero_padding_pl2(p_b_as_pl2)
                    
                    take = self.get_min_take(p_a_as_pl1, p_b_as_pl2) # A = pl.1, B = pl.2
                    pay_pl[i,j] = take @ self.payoffs_pl1
                    
                elif player == 2:
                    p_b_as_pl1 = strategy_b[:self.nb_actions]
                    assert np.isclose(p_b_as_pl1.sum(), 1.)
                    p_b_as_pl1 = self.zero_padding_pl1(p_b_as_pl1)
                    
                    take = self.get_min_take(p_b_as_pl1, p_a_as_pl2) # A = pl.2, B = pl.1
                    pay_pl[i,j] = take @ self.payoffs_pl2
                
        return pay_pl
        
    
    def calculate_payoffs(self) -> np.ndarray:
        
        self.payoffs_ = .5 * (self.calculate_payoffs_pl(player=1) + self.calculate_payoffs_pl(player=2))

        return self.payoffs()
    
    
    def get_normal_form(self) -> np.ndarray:
        A = np.zeros((self.nb_actions, self.nb_actions), dtype = float)
        B = np.zeros((self.nb_actions, self.nb_actions), dtype = float)
        
        for i in range(self.nb_actions):
            for j in range(self.nb_actions):
                if j>=i:
                    A[i,j] = self.payoffs_pl1[2*i]
                    B[i,j] = self.payoffs_pl2[2*i]
                else:
                    A[i,j] = self.payoffs_pl1[2*j + 1]
                    B[i,j] = self.payoffs_pl2[2*j + 1]
                    
        return A, B
        
        
        
        
        
        
        
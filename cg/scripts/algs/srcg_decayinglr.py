"""
    srcg: constant learning rate with no exponential smoothing
"""
from cg.scripts.algs.base_srcg import base_srcg


class srcg_decayinglr(base_srcg):
    
    def __init__(self,train_data,train_class,test_data,test_class,df,df_test,
                      distance="euclidian",stopping_condition=None,
                      stopping_percentage=None,lr=None,
                      selected_col_index=0,scale=True):
        
     
        
        super(srcg_decayinglr,self).__init__(train_data,train_class,test_data,test_class,df,df_test,
                          distance=distance,stopping_condition=stopping_condition,
                          stopping_percentage=stopping_percentage,lr=lr,
                          selected_col_index=0,scale=scale)
    
    
    def reference_weights(self):
        return self.weight_record[-1] 
    
    def schedule_lr(self):
        self.lr=self.lr_init*(self.counter**0.5)

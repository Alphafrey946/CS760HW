import numpy as np
import random

class Decisiontree(object):
  def __init__(self, data):
    ## since the feature is R^2, don't need to save the header 
    #self.data = np.zeros((data.shape[0]+1,data[:,:].shape[1]))
    #self.column_num = data[:,:-1].shape[1] # assume the last column is the label
    #self.column = [i for i in range(self.column_num) ]
    #self.data[1:,:] = data
    #self.data[0,:-1] = self.column
    #self.data[0,-1] = -1

    self.column_num = data[:,:-1].shape[1] # assume the last column is the label
    self.column = [i for i in range(self.column_num) ]
    self.data = data
    self.tree = None
    self.node = 0

  def _entropy(self,label):
    p0 = len(label[label==0])/len(label)
    p1 = len(label[label==1])/len(label)
    if p0 >0 and p1 >0:
       ent = -((p0)*np.log2(p0) + (p1)*np.log2(p1))
    elif p0 ==0:
      ent =  (p1)*np.log2(p1)
    elif p1 ==0:
       ent =  (p0)*np.log2(p0)
    #print(p0,p1,ent)
    return ent
  def _MutualInformaiton(self,D,V,c,column):
    left_ = V[V <c]
    right_ = V[V>=c]
    #label = D[1:,-1]
    label = D[:,-1]
    ent = self._entropy(label)
    #D = D[1:,:]
    D_left_ = D[V<c,-1]
    D_right_ =  D[V>=c,-1]

    if len(D_right_)>0 and len(D_left_)>0:
      ent_left_ = self._entropy(D_left_)
      ent_right_ = self._entropy(D_right_)

    elif len(D_right_)==0:
      ent_right_ = 0
      ent_left_ = self._entropy(D_left_)
    elif len(D_left_)==0:
      ent_right_ = self._entropy(D_right_)
      ent_left_ = 0
    return ent - ((len(left_)/D.shape[0])*ent_left_+(len(right_)/D.shape[0])*ent_right_)

  def _DetermineCandidateNumericSplits(self,D,V,column):
    MI_max = -np.inf
    thres=0
    for i in range(len(V)):
            c = V[i]
            MI_value = self._MutualInformaiton(D,V,c,column)
            #s_ratio = self._entropy(D[1:,-1])
            s_ratio = self._entropy(D[:,-1])
            #print(column,c,MI_value)
            if s_ratio ==0: # skip those candidate splits with zero split information
              continue
            MIR = MI_value/s_ratio
            if MIR > MI_max:
              # If there is a tie, then the last candidate would be split through 
              MI_max = MIR
              thres = c
    return thres
  def _DetermineCandidateSplits(self, D):
    label = D[:,-1]
    if len(label[label==1])==0:
      C = [0]
      return C
    elif len(label[label==0])==0:
      C = [1]
      return C
    
    else:
      results_ = np.zeros((len(self.column),3))
      for column in self.column:
            #V=D[1:,column]
            V=D[:,column]
            thres = self._DetermineCandidateNumericSplits(D, V,column)
            MI_ = self._MutualInformaiton(D,V,thres,column)
            #s_ratio = self._entropy(D[1:,-1])
            s_ratio = self._entropy(D[:,-1])
            #print(s_ratio)
            if s_ratio ==0: # skip those candidate splits with zero split information
              continue
            C_ = [thres,column,MI_/s_ratio]
            results_[column,:] = np.asarray(C_)
      if results_[0,2] >=results_[1,2]:
        return results_[0,:].tolist()
      elif results_[0,2] <results_[1,2]:
        return results_[1,:].tolist()   


  def _MakeSubtree(self,D):
    C = self._DetermineCandidateSplits(D)
    #print(D,C)
    if len(C)==1 or C[2]==0:
      # reach stopping criteria
      # node is empty
      # all splits have zero gain ratio
      # entropy of any candidate split is zero
      # create a leaf note
      self.node+=1
      return {'leaf':True,'Pred':C[0]}
    else:
      #D_top = D[0,:]
      #temp_tree = D[1:,C[1]]
      #print(D[:,int(C[1])])
      temp_tree = D[:,int(C[1])]
      left_tree_ = D[temp_tree < C[0]]
      right_tree_ = D[temp_tree >=C[0]]
      self.node+=1
      left_tree = self._MakeSubtree(left_tree_)
      right_tree = self._MakeSubtree(right_tree_)

      #return node
      #return {'leaf':False,'Threshold':C,'Left':left_tree,'Left_data': left_tree,'Right':right_tree,'Right_data': right_tree_}
      return {'leaf':False,'Threshold':C,'Left':left_tree,'Right':right_tree}
  


  def __call__(self):
    finaltree_ = self._MakeSubtree(self.data)
    self.tree = finaltree_

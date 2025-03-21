import numpy as np
import csv 
from sklearn.model_selection import train_test_split


file_name = '/data/haominpeng/Work/dian/exam/random forest/iris.csv'
f = open(file_name, 'r')
csvreader = csv.reader(f)
data_list = list(csvreader)
del data_list[-1]
#print(data_list[-1])

class2idx={}
idx2class={}

for i in range(len(data_list)):
    class_name = data_list[i][4]
    if class_name not in class2idx:
        class2idx[class_name] = len(class2idx)
        idx2class[len(idx2class)] = class_name
#print(class2idx)
#print(idx2class)

data_arr=np.array(data_list)
for i in range(len(data_list)):
    data_arr[i][4] = class2idx[data_list[i][4]]

#print(data_arr)
X = data_arr[:,0:4].astype(float)
y= data_arr[:,4].astype(int)

'''
1-1 逐步完成随机森林模型的构建
'''

'''
决策树的构建:CART决策树
'''
class DecisionTree:
    def __init__(self,max_depth=3, min_samples_split=2, features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = features if features is not None else range(X.shape[1])
        self.tree = None


    def gini(self, y):
        y=y.astype(int)
        counts = np.bincount(y)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    '''
    寻找最佳特征和最佳特征值
    '''
    def find_feature_value(self, X, y):
        # print('X:',X)
        # print('y:',y)
        best_gini = 3.3
        best_feature = None
        best_value = None
        #print('features:',self.features)
        for feature in self.features:
            #print('values:',np.unique(X[:,feature]))
            values = np.unique(X[:, feature])
            for value in values:
                y_l = y[X[:, feature] < value]
                y_r = y[X[:, feature] >= value]
                #print(len(y_l),self.min_samples_split)
                if len(y_l) < self.min_samples_split or len(y_r) < self.min_samples_split:
                    continue
                gini_l = self.gini(y_l)
                gini_r = self.gini(y_r)
                gini_f = gini_l * len(y_l) / len(y) + gini_r * len(y_r) / len(y)
                #print('gini_f:',gini_f,'best_gini:',best_gini)
                if gini_f < best_gini:
                    best_gini = gini_f
                    best_feature = feature
                    best_value = value
            
        return best_feature, best_value
    
    def _build_tree(self, X, y, depth):
        feature, value = self.find_feature_value(X, y)
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.argmax(np.bincount(y))
        if len(self.features) == 0:
            return np.argmax(np.bincount(y))
        if feature is None or value is None:
            return np.argmax(np.bincount(y))
        if len(np.unique(y)) == 1:
            return y[0]
        #print(type(X[:, feature]),type(value),'feature:',feature,'value:',value)
        left_idx = X[:, feature] < value
        right_idx = X[:, feature] >= value
        node = {
            'feature': feature,
            'value': value,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }
        return node
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)
    
    def _predict(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature']] < tree['value']:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])

'''
随机森林的构建
'''        
class RandomForest:
    def __init__(self, n_tree=100, max_depth=3, min_samples_split=2, max_features=None,max_samples=None):
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features 
        self.trees = []
        self.max_samples = max_samples

#bfs遍历所有树，统计每个特征被选取的次数，反映其重要性
    def compute_feature_importance(self):
        n_features = len(self.trees[0].features) if self.trees else 0
        self.feature_importances_ = np.zeros(X.shape[1])
        
        for tree in self.trees:
            stack = [tree.tree]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    #print(self.feature_importances_)
                    global_feature = node['feature']
                    self.feature_importances_[global_feature] += 1
                    stack.append(node['left'])
                    stack.append(node['right'])
        print(self.feature_importances_)
        self.feature_importances_ /= np.sum(self.feature_importances_)
        return self.feature_importances_
    
    def fit(self, X, y):
        for i in range(self.n_tree):
            #Bootstrap样本采样
            idx=np.random.choice(X.shape[0], self.max_samples, replace=True)
            Xs=X[idx]
            ys=y[idx]
            #特征子集采样
            if self.max_features:
                features = np.random.choice(Xs.shape[1], self.max_features, replace=False)
            else:
                features = np.random.choice(Xs.shape[1], max_features = int(np.sqrt(Xs.shape[1])) , replace=False)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, features=features)
            tree.fit(Xs, ys)
            self.trees.append(tree)
#投票预测实现   
    def predict(self, X):
        y_preds = np.array([tree._predict(X, tree.tree) for tree in self.trees])
        return np.argmax(np.bincount(y_preds))
    
'''
1-2 模型训练与模型评估
'''
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
rf = RandomForest(n_tree=100, max_depth=3, min_samples_split=2, max_features=2,max_samples=100)
rf.fit(X_train, y_train)

y_pred = np.array([rf.predict(x) for x in X_test])
print('Accuracy:', np.mean(y_pred == y_test))

'''
1-3 特征重要性评估
'''

importance=rf.compute_feature_importance()

idx2feature={0:'sepal length in cm',1:'sepal width in cm',2:'petal length in cm',3:'petal width in cm'}
for i in range(len(importance)):
    print('importance of ',idx2feature[i],' is ',importance[i])
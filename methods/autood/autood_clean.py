import numpy as np
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
import os
import csv
import math


learning_rate = 0.01
log_interval = 10

class Net(nn.Module):
    def __init__(self, dim = 10, args=None):
        super(Net, self).__init__()
        self.args = args
    	
        if args.batch_norm == "true":
            self.fc1 = nn.Linear(dim, 50)
            self.bn1 = nn.BatchNorm1d(50)
            self.fc2 = nn.Linear(50, 100)
            self.bn2 = nn.BatchNorm1d(100)
            self.fc3 = nn.Linear(100, 50)
            self.bn3 = nn.BatchNorm1d(50) 
            self.fc4 = nn.Linear(50, 2)
        else:
            self.fc1 = nn.Linear(dim, 50)
            self.fc2 = nn.Linear(50, 100)
            self.fc3 = nn.Linear(100,50)
            self.fc4 = nn.Linear(50,2)
    def forward(self, x):
        if self.args.batch_norm == "true":
            out = F.relu(self.bn1(self.fc1(x)))
            out = F.relu(self.bn2(self.fc2(out)))
            out = F.relu(self.bn3(self.fc3(out)))
            out = self.fc4(out)
            return F.log_softmax(out, -1)
        else:
            out = F.relu(self.fc1(x))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = self.fc4(out)
            return F.log_softmax(out, -1)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def get_device():
	if torch.cuda.is_available():  
		print("GPU detected, use gpu")
		dev = "cuda:0" 
	else:  
		dev = "cpu" 
	return dev 


def get_device():
	if torch.cuda.is_available():  
		print("GPU detected, use gpu")
		dev = "cuda:0" 
	else:  
		dev = "cpu" 
	return dev 


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def run_NN(X,y, epochs = 3,  dim = 10, train_batch_size=100,eval_batch_size=1, return_loss=False, args=None):
	dev = get_device()
	device = torch.device(dev)
	net = Net(dim=dim, args=args)
	net = net.to(device)
	# create a stochastic gradient descent optimizer
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
	# create a loss function
	criterion = nn.NLLLoss()
	# create dataset
	tensor_x = torch.tensor(X, device=device) # transform to torch tensor
	tensor_y = torch.tensor(y,dtype=torch.long, device=device)
	my_dataset = data.TensorDataset(tensor_x,tensor_y) # create your dataset
	
	if len(my_dataset) % 100 == 1 and args.batch_norm == 'true':
		train_batch_size = train_batch_size-1

	train_dataloader = data.DataLoader(my_dataset, batch_size=train_batch_size, shuffle = True) # create your dataloader

	# run the main training loop
	for epoch in range(epochs):
		net.train()
		for batch_idx, (input_data, target) in enumerate(train_dataloader):
			input_data, target = Variable(input_data), Variable(target)
			#input_data = input_data.to(device)
			#target = input_data.to(device)
			net_out = net(input_data.float())
			loss = criterion(net_out, target.long())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		net.eval()
		losses = AverageMeter()
		top1 = AverageMeter()
		top2 = AverageMeter()

	if return_loss:
		net.eval()
		criterion = nn.NLLLoss(reduction='none')
	 	
		top1 = AverageMeter()
		top2 = AverageMeter()
		loss_list = []
		for batch_idx, (input_data, target) in enumerate(data.DataLoader(my_dataset, batch_size=eval_batch_size, shuffle=False)):
			input_data, target = Variable(input_data), Variable(target)
			net_out = net(input_data.float())
			loss = criterion(net_out, target)
			prec = accuracy(net_out.data, target)
			loss_list.append(loss.data.cpu().numpy())
			top1.update(prec[0], input_data.size(0))
		print('Final Training Result: ' 'Prec @ 1 {top1.avg:.3f}%'.format(top1=top1))   
		return np.concatenate(loss_list,axis=0), net
	else:
		return None, net

def inference_NN(net, testing_X, testing_y = None):
	dev = get_device()
	device = torch.device(dev)
	test_dataloader = data.DataLoader(data.TensorDataset(torch.tensor(testing_X,device=device), torch.tensor(testing_y,device=device)), batch_size=100, shuffle=False) 
	net.eval()
	predict_proba = []
	for batch_idx, (input_data, target) in enumerate(test_dataloader):
		input_data = Variable(input_data)
		net_out = net(input_data.float())
		predict_proba.append(F.softmax(net_out, dim=1).data.cpu().numpy())
	return np.concatenate(predict_proba)


def run_autood_clean(X, y, L, pred_labels, args, ratio_to_remove=0.05, max_iteration=15, separate_inline_outlier=True, show_metrics=True, inlier=0.0001, outlier=0.9999, early_stop=True):
	
	torch.manual_seed(args.seed)
	ratio_to_remove = ratio_to_remove
	dim = np.shape(X)[1]
	remain_points = np.array(range(len(y)))
	
	if args.label_noise == "autood_ensemble_mv":
		mid = np.shape(L)[1]/2
		label_of_point = np.full((len(y)), 0)
		label_of_point[np.sum(L, axis = 1) > mid] = 1
	else:
		label_of_point = pred_labels

	# transformer = RobustScaler().fit(X)
	X_transformed = X

	prev_loss = 10000

	for i_range in range(0, max_iteration):
		print("##################################################################")
		print('Iteration = {}'.format(i_range))

		# if((i_range  + 1) % 1 == 0) and show_metrics:
		# 	clf_X = SVC(gamma='auto', probability=True, random_state=0)
		# 	clf_X.fit(X_transformed[remain_points], label_of_point[remain_points])
		# 	clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:,1]

		# 	SVM_threshold = 0.5
		# 	print("F-1 score from SVM:",metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 	print("precision from SVM:",metrics.precision_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 	print("recall from SVM:",metrics.recall_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 	SVM_threshold = np.sort(clf_predict_proba_X)[::-1][int(np.sum(y))]
		# 	print("F-1 score from SVM:",metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 	print("precision from SVM:",metrics.precision_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 	print("recall from SVM:",metrics.recall_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))

		temp_remain_points = remain_points.copy()
	    # start pruning points
		loss_list, model = run_NN(X_transformed[temp_remain_points],label_of_point[temp_remain_points], 3, dim = dim, train_batch_size=100, eval_batch_size=100, return_loss=True, args=args)

		if not separate_inline_outlier:
			# Do not separate inlier and outlier
			loss_threshold = np.sort(loss_list)[::-1][int(ratio_to_remove * len(loss_list))]
			# print(min(loss_list), max(loss_list), loss_threshold, np.mean(loss_list)+ np.std(loss_list))
			loss_threshold = np.mean(loss_list)+ np.std(loss_list)
			points_to_remove = temp_remain_points[(loss_list > loss_threshold)]

		else:
			# separate inline and outlier
			inlier_labels = np.where(label_of_point[temp_remain_points] == 0)[0]  # MV predicted inliers
			#loss_threshold = np.sort(loss_list[inlier_labels])[::-1][int(ratio_to_remove * len(loss_list[inlier_labels]))]
			loss_threshold = np.mean(loss_list[inlier_labels])+ np.std(loss_list[inlier_labels])
			points_to_remove = temp_remain_points[inlier_labels][(loss_list[inlier_labels] > loss_threshold)]

			outlier_labels = np.where(label_of_point[temp_remain_points] == 1)[0]
			#loss_threshold = np.sort(loss_list[outlier_labels])[::-1][int(ratio_to_remove * len(loss_list[outlier_labels]))]
			loss_threshold = np.mean(loss_list[outlier_labels])+ np.std(loss_list[outlier_labels])
			points_to_remove = np.append(points_to_remove, temp_remain_points[outlier_labels][(loss_list[outlier_labels] > loss_threshold)])

		_, model = run_NN(X_transformed[temp_remain_points],label_of_point[temp_remain_points],10, dim = dim, train_batch_size=100, eval_batch_size=100, return_loss=False, args=args)
		predict_proba = inference_NN(model, X_transformed, y)[:,1]
		if show_metrics:
			print("F-1 score from NN:",metrics.f1_score(y, np.array([int(i) for i in predict_proba > 0.5])))
			print("Precision score from NN:",metrics.precision_score(y, np.array([int(i) for i in predict_proba > 0.5])))
			print("Recall score from NN:",metrics.recall_score(y, np.array([int(i) for i in predict_proba > 0.5])))
			print(f"ROC AUC score from NN: {metrics.roc_auc_score(y, predict_proba)}")

		print('Number of points to remove: ', len(points_to_remove))
		temp_remain_points = np.setdiff1d(np.array(temp_remain_points), points_to_remove)

		predict_outlier_indexes = np.where(predict_proba > outlier)[0]
		new_outlier_indexes = np.setdiff1d(predict_outlier_indexes, temp_remain_points)

		if(len(new_outlier_indexes) > 0):
			print('F-1 before: ', metrics.f1_score(y[new_outlier_indexes], label_of_point[new_outlier_indexes]))
			label_of_point[new_outlier_indexes] = 1
			print('F-1 after: ', metrics.f1_score(y[new_outlier_indexes], label_of_point[new_outlier_indexes]))
			temp_remain_points = np.union1d(temp_remain_points, predict_outlier_indexes)

		predict_inlier_indexes = np.where(predict_proba < inlier)[0]
		new_inlier_indexes = np.setdiff1d(predict_inlier_indexes, temp_remain_points)

		if(len(new_inlier_indexes) > 0):
			label_of_point[new_inlier_indexes] = 0
			temp_remain_points = np.union1d(temp_remain_points, predict_inlier_indexes)

		# if(early_stop and len(new_outlier_indexes) + len(new_inlier_indexes) > len(points_to_remove)):# or len(remain_points) < np.shape(L)[0]/3):  # 
		# 	clf_X = SVC(gamma='auto', probability=True, random_state=0)
		# 	clf_X.fit(X_transformed[remain_points], label_of_point[remain_points])
		# 	clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:,1]

		# 	if show_metrics:
		# 		SVM_threshold = 0.5
		# 		print("F-1 score from SVM:",metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 		print("precision from SVM:",metrics.precision_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 		print("recall from SVM:",metrics.recall_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 		SVM_threshold = np.sort(clf_predict_proba_X)[::-1][int(np.sum(y))]
		# 		print("F-1 score from SVM:",metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 		print("precision from SVM:",metrics.precision_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 		print("recall from SVM:",metrics.recall_score(y, np.array([int(i) for i in clf_predict_proba_X > SVM_threshold])))
		# 	break
		# else:
		remain_points = temp_remain_points

	
	# get final metrics
	f1_score = metrics.f1_score(y, np.array([int(i) for i in predict_proba > 0.5]))
	precision = metrics.precision_score(y, np.array([int(i) for i in predict_proba > 0.5]))
	recall = metrics.recall_score(y, np.array([int(i) for i in predict_proba > 0.5]))
	auc = metrics.roc_auc_score(y, predict_proba)
	
	# write results to csv
	if args.experiment == "varying_noise_rate":
		csv_file_path = f"results/final_results/{args.experiment}_analysis/{args.experiment}_{args.dataset}.csv"
		path_exists = False
		if os.path.exists(csv_file_path):
			path_exists = True
		with open(csv_file_path, mode='a', newline='') as file:
			csv_writer = csv.writer(file)
			if not path_exists:
				csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "epoch", 
									"agree_f1", "agree_n", "agree_precision", "agree_recall",
									"model1_disagree_f1", "model1_disagree_n", "model1_disagree_precision", "model1_disagree_recall",
									"model2_disagree_f1", "model2_disagree_n", "model2_disagree_precision", "model2_disagree_recall",
									"model1_ctl_f1", "model1_ctl_n", "model1_ctl_precision", "model1_ctl_recall",
									"model2_ctl_f1", "model2_ctl_n", "model2_ctl_precision", "model2_ctl_recall",
									"overall_f1", "overall_precision", "overall_recall", "overall_auc",
									"agree_time", "disagree_time", "refurbishment_time", "training_time"])
				
			csv_writer.writerow([args.method, args.dataset, args.seed, args.batch_norm, args.epochs,
								0, 0, 0, 0,
								0, 0, 0, 0,
								0, 0, 0, 0,
								0, 0, 0, 0,
								0, 0, 0, 0,
								f1_score, precision, recall, auc,
								0, 0, 0, 0])

	else:
		csv_file_path = f"results/final_results/sota_comparison/results.csv"
		path_exists = False
		if os.path.exists(csv_file_path):
			path_exists = True
		
		with open(csv_file_path, mode='a', newline='') as file:
			csv_writer = csv.writer(file)
			if not path_exists:
				csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "f1", "precision", "recall", "auc"])
			csv_writer.writerow([args.method, args.dataset, args.seed, args.batch_norm, f1_score, precision, recall, auc])

		








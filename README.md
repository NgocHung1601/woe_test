# woe_test


# Create a scaler object
std_slc = StandardScaler()

# Create a PCA object
pca = decomposition.PCA()

# Create a logistic Regression with an L2 penalty
logistic_Reg = LogisticRegression()

# Create a pipeline for 3 steps
# first, standardlize data
# second, transform data with PCA
# Third, train LR model
pipe = Pipeline(steps=[('std_slc', std_slc),
                        ('pca', pca),
                        ('logistic_Reg', logistic_Reg)])

# Create parameters space
n_components = list(range(1,X.shape[1]+1,1))
C = np.logspace(-4, 4, 50)
penalty = ['l1', 'l2']

parameters = dict(pca__n_components=n_components,
                logistic_Reg__C=C,
                logistic_Reg__penalty=penalty)

# Create a gridsearch object
clf_GS = GridSearchCV(pipe, parameters)

# Fit the gridsearch
clf_GS.fit(x_train, y_train)

# View best parameters
print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['logistic_Reg'])

# Use cross validation to evaluate model
CV_result = cross_val_score(clf, x_train, y_train, cv = 10, n_jobs = -1)
print(CV_result, CV_result.mean(), CV_result.std())

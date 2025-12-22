### AUC performance comparison between two methods using multiple iterations 
#of true labels and corresponding predicted labels.


import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


'''
Generate synthetic label data.
'''
def generate_synthetic_label_data():
    predicted_labels_existing = []
    predicted_labels_proposed = []

    #Generate synthetic predicted labels from an existing method.
    #20 runs and each run has 20 labels.
    #The synthetic data is stored in a 2D list.
    for i in range(20):

        predicted_labels_p1 = []
        predicted_labels_p2 = []
        for idx in range(20):

            probability_of_true = 0.5
            probability_of_false = 1.0 - probability_of_true

            ran_bool = random.choices([True, False], weights=[probability_of_true
                                                              , probability_of_false])[0]
            temp_num = 0
            if ran_bool:
                temp_num = random.uniform(0.9, 1.0)
            else:
                temp_num = random.uniform(0.0, 0.1)
            predicted_labels_p1.append(temp_num)

            ran_bool = random.choices([True, False], weights=[probability_of_true
                                                              , probability_of_false])[0]
            if ran_bool:
                temp_num = random.uniform(0.0, 0.1)
            else:
                temp_num = random.uniform(0.9, 1.0)
            predicted_labels_p2.append(temp_num)

        predicted_labels_temp = predicted_labels_p1 + predicted_labels_p2
        predicted_labels_existing.append(predicted_labels_temp)


    #Generate synthetic predicted labels from the proposed method.
    #20 runs and each run has 20 labels.
    #The synthetic data is stored in a 2D list.
    for i in range(20):

        predicted_labels_p1 = []
        predicted_labels_p2 = []
        for idx in range(20):

            probability_of_true = 0.9
            probability_of_false = 1.0 - probability_of_true

            ran_bool = random.choices([True, False], weights=[probability_of_true
                                                              , probability_of_false])[0]
            temp_num = 0
            if ran_bool:
                temp_num = random.uniform(0.9, 1.0)
            else:
                temp_num = random.uniform(0.0, 0.1)
            predicted_labels_p1.append(temp_num)

            ran_bool = random.choices([True, False], weights=[probability_of_true
                                                              , probability_of_false])[0]
            if ran_bool:
                temp_num = random.uniform(0.0, 0.1)
            else:
                temp_num = random.uniform(0.9, 1.0)
            predicted_labels_p2.append(temp_num)

        predicted_labels_temp = predicted_labels_p1 + predicted_labels_p2
        predicted_labels_proposed.append(predicted_labels_temp)

    return predicted_labels_existing, predicted_labels_proposed


'''
Plot an average AUC figure.
'''
def plot_average_auc_from_multiple_runs(ax, true_labels, predicted_labels_mat
                        , method_name="existing method", color_to_method="red"):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for index,predicted_labels in enumerate(predicted_labels_mat):

        curve_parameters = {'color': color_to_method, 'label': None, 'alpha':0.3
                            , 'lw':1}
        viz = metrics.RocCurveDisplay.from_predictions(true_labels, predicted_labels
                                              ,ax=ax, curve_kwargs=curve_parameters)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color=color_to_method,
        label=f"Average ROC (AUC = {mean_auc:.2f}) from {method_name}",
        lw=2,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color=color_to_method,
        alpha=0.1,
    )
    ax.plot([0, 0], [0, 0], lw=1, color=color_to_method
            , label=f"ROCs from {method_name}", alpha=0.2)


predicted_labels_existing, predicted_labels_proposed = generate_synthetic_label_data()

fig, ax = plt.subplots(figsize=(10,12))

#True label list
true_labels = [1,1,1,1,1,1,1,1,1,1
              ,1,1,1,1,1,1,1,1,1,1
              ,0,0,0,0,0,0,0,0,0,0
              ,0,0,0,0,0,0,0,0,0,0
              ]

#Plot an average AUC line and other stats from multiple AUCs for an existing method.
method_name = "existing method"
color="grey"
plot_average_auc_from_multiple_runs(ax, true_labels, predicted_labels_existing
                                    , method_name=method_name, color_to_method=color)

#Plot an average AUC line and other stats from multiple AUCs for the proposed method.
method_name = "proposed method"
color="red"
plot_average_auc_from_multiple_runs(ax, true_labels, predicted_labels_proposed
                                    , method_name=method_name, color_to_method=color)

#Plot a line for random guess
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Random guess"
        , alpha=0.8)


ax.set(xlim=[-0.05, 1.05],ylim=[-0.05, 1.05],)
plt.xlabel("False positive rate",fontsize=20)
plt.ylabel("True positive rate",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#Setrings for legends
ax.legend(loc="lower right",fontsize=12,frameon=False)
#Display the figure
plt.show()


class FairClassifier(object):
    def __init__(self, n_features, n_sensitive, lambdas):
        self.lambdas = lambdas

        clf_inputs = Input(shape=(n_features,))
        adv_inputs = Input(shape=(1,))

        clf_net = self._create_clf_net(clf_inputs)
        adv_net = self._create_adv_net(adv_inputs, n_sensitive)
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(adv_net)
        self._clf = self._compile_clf(clf_net)
        self._clf_w_adv = self._compile_clf_w_adv(clf_inputs, clf_net, adv_net)
        self._adv = self._compile_adv(clf_inputs, clf_net, adv_net, n_sensitive)
        self._val_metrics = None
        self._fairness_metrics = None

        self.predict = self._clf.predict

    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag

        return make_trainable

    def _create_clf_net(self, inputs):
        dense1 = Dense(32, activation="relu")(inputs)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation="relu")(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation="relu")(dropout2)
        dropout3 = Dropout(0.2)(dense3)
        outputs = Dense(1, activation="sigmoid", name="y")(dropout3)
        return Model(inputs=[inputs], outputs=[outputs])

    def _create_adv_net(self, inputs, n_sensitive):
        dense1 = Dense(32, activation="relu")(inputs)
        dense2 = Dense(32, activation="relu")(dense1)
        dense3 = Dense(32, activation="relu")(dense2)
        outputs = [Dense(1, activation="sigmoid")(dense3) for _ in range(n_sensitive)]
        return Model(inputs=[inputs], outputs=outputs)

    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        clf.compile(loss="binary_crossentropy", optimizer="adam")
        return clf

    def _compile_clf_w_adv(self, inputs, clf_net, adv_net):
        clf_w_adv = Model(
            inputs=[inputs], outputs=[clf_net(inputs)] + adv_net(clf_net(inputs))
        )
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.0] + [-lambda_param for lambda_param in self.lambdas]
        clf_w_adv.compile(
            loss=["binary_crossentropy"] * (len(loss_weights)),
            loss_weights=loss_weights,
            optimizer="adam",
        )
        return clf_w_adv

    def _compile_adv(self, inputs, clf_net, adv_net, n_sensitive):
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        adv.compile(loss=["binary_crossentropy"] * n_sensitive, optimizer="adam")
        return adv

    def _compute_class_weights(self, data_set):
        class_values = [0, 1]
        class_weights = []
        if len(data_set.shape) == 1:
            balanced_weights = compute_class_weight("balanced", class_values, data_set)
            class_weights.append(dict(zip(class_values, balanced_weights)))
        else:
            n_attr = data_set.shape[1]
            for attr_idx in range(n_attr):
                balanced_weights = compute_class_weight(
                    "balanced", class_values, np.array(data_set)[:, attr_idx]
                )
                class_weights.append(dict(zip(class_values, balanced_weights)))
        print(f"-compute-target-class_weights > {class_weights}")
        return class_weights

    def _compute_target_class_weights(self, y):
        class_values = [0, 1]
        balanced_weights = compute_class_weight("balanced", class_values, y)
        class_weights = {"y": dict(zip(class_values, balanced_weights))}
        print(f"-compute-class-weights > {class_weights}")
        return class_weights

    def pretrain(self, x, y, z, epochs=10, verbose=0):
        self._trainable_clf_net(True)
        self._clf.fit(x.values, y.values, epochs=epochs, verbose=verbose)
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        class_weight_adv = self._compute_class_weights(z)
        print(f"pretrain-class-weight-adv > {class_weight_adv}")
        self._adv.fit(
            x.values,
            np.hsplit(z.values, z.shape[1]),
            class_weight=class_weight_adv,
            epochs=epochs,
            verbose=verbose,
        )

    def fit(
        self, x, y, z, validation_data=None, T_iter=250, batch_size=128, save_figs=False
    ):
        n_sensitive = z.shape[1]
        if validation_data is not None:
            x_val, y_val, z_val = validation_data

        class_weight_adv = self._compute_class_weights(z)
        class_weight_clf_w_adv = [{0: 1.0, 1: 1.0}] + class_weight_adv
        print(f" class_weight_clf_w_adv-- {class_weight_clf_w_adv}")
        self._val_metrics = pd.DataFrame()
        self._fairness_metrics = pd.DataFrame()
        for idx in range(T_iter):
            if validation_data is not None:
                self._clf.save(
                    f"output/models/{idx:03d}.h5"
                )  # creates a HDF5 file 'my_model.h5'
                y_pred = pd.Series(self._clf.predict(x_val).ravel(), index=y_val.index)
                (
                    thresholdOpt,
                    metricOpt,
                    rscoreOpt,
                    pscoreOpt,
                    srscoreOpt,
                ) = return_metric(y_pred, z_val)

                print(f"Model= {idx}")

                self._val_metrics.loc[idx, "model"] = idx
                self._val_metrics.loc[idx, "thresholdOpt"] = thresholdOpt
                self._val_metrics.loc[idx, "metricOpt"] = metricOpt
                self._val_metrics.loc[idx, "rscoreOpt"] = rscoreOpt
                self._val_metrics.loc[idx, "pscoreOpt"] = pscoreOpt
                self._val_metrics.loc[idx, "srscoreOpt"] = srscoreOpt

                # pyplot.plot(recall_prc, precision_prc, marker='.', label='NN')
                # pyplot.scatter(recall_prc[f1_ix], precision_prc[f1_ix], marker='o', color='black', label='Best f1')
                # pyplot.scatter(recall_prc[recall_ix], precision_prc[recall_ix], marker='o', color='orange', label='Best Recall')
                # axis labels
                # pyplot.xlabel('Recall')
                # pyplot.ylabel('Precision')
                # pyplot.legend()
                ## show the plot
                # pyplot.show()

            self._val_metrics = self._val_metrics.sort_values(
                by=["metricOpt"], ascending=False
            )
            self._val_metrics.to_csv("output/_val_metrics.csv")

            # train adverserial
            self._trainable_clf_net(False)
            self._trainable_adv_net(True)
            self._adv.fit(
                x.values,
                np.hsplit(z.values, z.shape[1]),
                batch_size=batch_size,
                class_weight=class_weight_adv,
                epochs=1,
                verbose=0,
            )

            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x))[:batch_size]
            self._clf_w_adv.train_on_batch(
                x.values[indices],
                [y.values[indices]] + np.hsplit(z.values[indices], n_sensitive),
                class_weight=class_weight_clf_w_adv,
            )


# HIDE
def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1 / odds]) * 100


def sr_rule(y_pred, threshold=0.5):
    BT = threshold
    df = pd.DataFrame(y_pred, columns=["raw"])
    df["raw_bin"] = df["raw"].apply(lambda x: 1 if x >= BT else 0)
    return df["raw_bin"].mean()


def return_metric(y_pred, y_val, z_val):
    thresholds = np.arange(0.0, 1.0, 0.001)
    rscore = np.zeros(shape=(len(thresholds)))
    pscore = np.zeros(shape=(len(thresholds)))
    srscore = np.zeros(shape=(len(thresholds)))

    # Fit the model
    for index, elem in enumerate(thresholds):
        # Corrected probabilities
        y_pred_prob = (y_pred > elem).astype("int")
        # Calculate the f-score
        rscore[index] = recall_score(y_val, y_pred_prob)
        pscore[index] = (p_rule(y_pred_prob, z_val["race"], threshold=elem)) / 100
        srscore[index] = 1 - abs(sr_rule(y_pred_prob, threshold=elem) - 0.50)

    rscore = np.nan_to_num(rscore)
    pscore = np.nan_to_num(pscore)
    srscore = np.nan_to_num(srscore)

    super_threshold_indices = srscore < 0.98
    srscore[super_threshold_indices] = 0

    metric = rscore * pscore * srscore

    index = np.argmax(metric)
    thresholdOpt = round(thresholds[index], ndigits=4)
    metricOpt = round(metric[index], ndigits=4)
    rscoreOpt = round(rscore[index], ndigits=4)
    pscoreOpt = round(pscore[index], ndigits=4)
    srscoreOpt = round(srscore[index], ndigits=4)

    print("Best Threshold: {} with Metric: {}".format(thresholdOpt, metricOpt))
    print("Best Threshold: {} with Recall: {}".format(thresholdOpt, rscoreOpt))
    print("Best Threshold: {} with Pvalue: {}".format(thresholdOpt, pscoreOpt))
    print("Best Threshold: {} with SR: {}".format(thresholdOpt, srscoreOpt))

    return thresholdOpt, metricOpt, rscoreOpt, pscoreOpt, srscoreOpt
#include <iostream>
#include <tuple>
#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp"

#include "cancer_analysis/util.hpp"

namespace cancer_analysis
{

struct SVMOptions
{
    double C = 0.1;
    double learning_rate = 0.001;
    unsigned epochs = 150;
};


class LinearSVM
{
public:
    LinearSVM() = default;

    std::tuple<std::vector<double>, xt::xarray<double>, double>
    fit(xt::xarray<double> X, xt::xarray<double> y, SVMOptions options = {});

    xt::xarray<double> predict(xt::xarray<double> X);

private:
    xt::xarray<double> w = xt::empty<double>({0});
    double b;

    double computeLoss(xt::xarray<double> X, xt::xarray<double> y, double C);
};

double LinearSVM::computeLoss(xt::xarray<double> X, xt::xarray<double> y, double C)
{
    if (w.shape()[0] == 0) {
        return -1.0;
    }
    xt::xarray<double> margins = y * (xt::linalg::dot(X, w) + b);
    return (xt::square(w) / 2 + C * xt::maximum(0, 1 - margins))(0);
}

std::tuple<std::vector<double>, xt::xarray<double>, double>
LinearSVM::fit(xt::xarray<double> X, xt::xarray<double> y, SVMOptions options)
{
    const auto& [C, lr, epochs] = options;
    w = xt::random::randn(std::vector<size_t>{X.shape()[1], 1}, 0.0, 1.0);
    b = 0;

    xt::xarray<double> margins, x_err, y_err;

    std::vector<double> losses;

    for (unsigned i = 0; i < epochs; ++i) {
        margins = y * (xt::linalg::dot(X, w) + b);
        auto idxs = xt::from_indices(xt::argwhere(margins < 1));

        x_err = xt::view(X, xt::keep(idxs), xt::all());
        y_err = xt::view(y, xt::keep(idxs));

        auto w_d = w - C * xt::linalg::dot(xt::transpose(x_err), y_err);
        auto b_d = -C * xt::sum(y_err)(0);   // note: xt::sum returns an xexpression, not a scalar

        w = w - lr * w_d;
        b -= lr * b_d;

        losses.push_back(computeLoss(X, y, C));
    }

    return {losses, w, b};
}

xt::xarray<double> LinearSVM::predict(xt::xarray<double> X)
{
    if (w.shape()[0] == 0) {
        return xt::empty<double>({0});
    }
    return xt::sign(xt::linalg::dot(X, w) + b);
}

} // namespace cancer_analysis

int main()
{
    xt::random::seed(80085);
    //xt::random::seed(1337);

    auto [train_obs, train_grp, test_obs, test_grp] =
        cancer_analysis::readData("data/ovariancancer_obs.csv", "data/ovariancancer_grp.csv");

    // TODO: Using the first three principle components while the SVM is in development.
    //       Will switch to the full dimensionality eventually.

    // Normalize the data and calculate the SVD
    xt::xarray<double> X = cancer_analysis::normalizeData(train_obs);
    xt::filtration(X, X < 0.5) = -1.0;  // change all the zeros to -1.0

    auto [U, S, VT] = xt::linalg::svd(X, false);
    // Use the first three principal components
    auto VTxyz = xt::eval(xt::view(VT, xt::range(0, 3), xt::all()));
    cancer_analysis::plotData(X, train_grp, VTxyz);

    cancer_analysis::LinearSVM svm;
    //xt::xarray<double> tmp = xt::linalg::dot(X, xt::transpose(VTxyz));
    //auto [losses, w, b] = svm.fit(tmp, train_grp);
    //auto svm_training_grp = svm.predict(tmp);
    ////cancer_analysis::plotData(X, svm_training_grp, VTxyz);
    //cancer_analysis::plotSVM(X, svm_training_grp, VTxyz, w, b);
    //cancer_analysis::plotLoss(losses);
    //auto svm_test_grp = svm.predict(xt::linalg::dot(Y, xt::transpose(VTxyz)));
    //cancer_analysis::plotSVM(Y, svm_test_grp, VTxyz, w, b);

    auto [losses, w, b] = svm.fit(X, train_grp);
    auto svm_training_grp = svm.predict(X);
    cancer_analysis::plotData(X, svm_training_grp, VTxyz);
    cancer_analysis::plotLoss(losses);

    xt::xarray<double> Y = cancer_analysis::normalizeData(test_obs);
    cancer_analysis::plotData(Y, test_grp, VTxyz);

    auto svm_test_grp = svm.predict(Y);
    cancer_analysis::plotData(Y, svm_test_grp, VTxyz);

    cancer_analysis::showPlots();

    return 0;
}

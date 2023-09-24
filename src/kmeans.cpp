#include <iostream>
#include <tuple>
#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor-blas/xlinalg.hpp"

#include "cancer_analysis/util.hpp"

namespace cancer_analysis
{

std::tuple<xt::xarray<double>, xt::xtensor_fixed<double, xt::xshape<2, 3>>>
computeKMeans(xt::xarray<double> observations,
              xt::xarray<double> VTxyz,
              unsigned max_iters)
{
    xt::xarray<double> points = xt::linalg::dot(observations, xt::transpose(VTxyz));

    size_t point_count = points.shape()[0];
    xt::xarray<double> groups = xt::ones<double>(std::vector<size_t>{point_count, 1});
    xt::xarray<double> old_groups = groups;

    xt::xarray<bool> mask;
    xt::xarray<double> tmp; // TODO: better name
    xt::xtensor_fixed<double, xt::xshape<1, 3>> k0, k1, p;
    // initialize cluster centers -- TODO: randomize
    k0 = xt::view(points, xt::range(2, 3), xt::all());
    k1 = xt::view(points, xt::range(8, 9), xt::all());

    for (unsigned i = 0; i < max_iters; ++i) {
        for (size_t j = 0; j < point_count; ++j) {
            p = xt::view(points, xt::range(j, j + 1), xt::all());
            if (xt::linalg::norm(k0 - p) < xt::linalg::norm(k1 - p)) {
                groups(j, 0) = 0.0;
            } else {
                groups(j, 0) = 1.0;
            }
        }

        if (groups == old_groups) break;
        old_groups = groups;

        mask = groups > 0.5;
        mask.reshape({-1, 1});
        tmp = xt::filter(points, xt::hstack(xt::xtuple(mask, mask, mask)));
        k0 = xt::mean(tmp.reshape({-1, 3}), 0);
        mask = groups < 0.5;
        mask.reshape({-1, 1});
        tmp = xt::filter(points, xt::hstack(xt::xtuple(mask, mask, mask)));
        k1 = xt::mean(tmp.reshape({-1, 3}), 0);
    }

    auto centers = xt::concatenate(xt::xtuple(k0, k1), 0);
    return {groups, centers};
}

xt::xarray<double> testKMeans(xt::xarray<double> observations,
                              xt::xarray<double> VTxyz,
                              xt::xtensor_fixed<double, xt::xshape<2, 3>> centers)
{
    xt::xarray<double> points = xt::linalg::dot(observations, xt::transpose(VTxyz));

    size_t point_count = points.shape()[0];
    xt::xarray<double> groups = xt::empty<double>(std::vector<size_t>{point_count, 1});

    xt::xtensor_fixed<double, xt::xshape<1, 3>> k0, k1, p;
    k0 = xt::view(centers, xt::range(0, 1), xt::all());
    k1 = xt::view(centers, xt::range(1, 2), xt::all());


    for (size_t i = 0; i < point_count; ++i) {
        p = xt::view(points, xt::range(i, i + 1), xt::all());
        if (xt::linalg::norm(k0 - p) < xt::linalg::norm(k1 - p)) {
            groups(i, 0) = 0.0;
        } else {
            groups(i, 0) = 1.0;
        }
    }

    return groups;
}

} // namespace cancer_analysis

int main()
{
    xt::random::seed(80085);
    //xt::random::seed(1337);

    auto [train_obs, train_grp, test_obs, test_grp] =
        cancer_analysis::readData("data/ovariancancer_obs.csv", "data/ovariancancer_grp.csv");

    // Normalize the data and calculate the SVD
    xt::xarray<double> X = cancer_analysis::normalizeData(train_obs);

    auto [U, S, VT] = xt::linalg::svd(X, false);
    // Use the first three principal components
    auto VTxyz = xt::view(VT, xt::range(0, 3), xt::all());
    cancer_analysis::plotData(X, train_grp, VTxyz);

    auto [kmeans_training_grp, centers]= cancer_analysis::computeKMeans(X, VTxyz, 100);
    cancer_analysis::plotData(X, kmeans_training_grp, VTxyz);

    auto error = xt::sum(xt::cast<int>(train_grp) ^ xt::cast<int>(kmeans_training_grp))();
    std::cout << "Number of misclassified patients on training set: " << error << std::endl;

    xt::xarray<double> Y = cancer_analysis::normalizeData(test_obs);

    auto kmeans_test_grp = cancer_analysis::testKMeans(Y, VTxyz, centers);
    cancer_analysis::plotData(Y, test_grp, VTxyz);
    cancer_analysis::plotData(Y, kmeans_test_grp, VTxyz);

    error = xt::sum(xt::cast<int>(test_grp) ^ xt::cast<int>(kmeans_test_grp))();
    std::cout << "Number of misclassified patients on test set: " << error << std::endl;

    cancer_analysis::showPlots();

    return 0;
}

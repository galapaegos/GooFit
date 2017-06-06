#pragma once

#include "goofit/PDFs/GooPdf.h"

namespace GooFit {


class ExpPdf : public GooPdf {
public:
    ExpPdf(std::string n, Variable* _x, Variable* alpha, Variable* offset = nullptr);
    ExpPdf(std::string n, Variable* _x, std::vector<Variable*>& weights, Variable* offset = nullptr);
    __host__ fptype integrate(fptype lo, fptype hi) const;
    __host__ virtual bool hasAnalyticIntegral() const {
        GOOFIT_TRACE("host_parameters[{}] = {}", parametersIdx, host_parameters[parametersIdx]);
        return (1 == host_parameters[parametersIdx]);
    }

    __host__ virtual void recursiveSetIndices ();

private:
    int ExpType;
};
} // namespace GooFit

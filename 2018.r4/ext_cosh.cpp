/*
// Copyright (c) 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class CoshImpl: public ExtLayerBase {
public:
    explicit CoshImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            addConfig(layer, {{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

		//hyperbolic cosine is given by :
		// (e^x + e^-x)/2
		parallel_for3d(N, C, H, [&](int b, int c, int h) {
        			// Fill output_sequences with -1
			for (size_t ii = 0; ii < b*c; ii++) {
				dst_data[ii] = (exp(src_data[ii]) + exp(-src_data[ii]))/2;
			}
        });
        return OK;
    }

private:
   
};

REG_FACTORY_FOR(ImplFactory<CoshImpl>, Cosh);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

/*
* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <string.h> // for basename(3) that doesn't modify its argument
#include <unistd.h> // for getopt
#include <vpi/Image.h>
#include <vpi/LensDistortionModels.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>

#include <iostream>
#include <sstream>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

int main(int argc, char *argv[])
{
    VPIImage input      = NULL;
    VPIImageFormat type;
    VPIImage output     = NULL;
    VPIStream stream    = NULL;
    int32_t w,h;
    VPIWarpMap map;
    VPIPayload warp;
    int retval = 0;

    try
    {
        if (argc != 3)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|vic|cuda> <input image>");
        }

        std::string strBackend       = argv[1];
        std::string strInputFileName = argv[2];

        cv::Mat cvImage = cv::imread(strInputFileName);
        if (cvImage.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileName + "'");
        }

        assert(cvImage.type() == CV_8UC3);

        VPIBackend backend;

        if (strBackend == "cpu")
        {
            backend = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backend = VPI_BACKEND_CUDA;
        }
        else if (strBackend == "vic")
        {
            backend = VPI_BACKEND_VIC;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend + "' not recognized, it must be either cpu, cuda or vic");
        }

        CHECK_STATUS(vpiStreamCreate(backend | VPI_BACKEND_CUDA, &stream));

        vpiImageCreateOpenCVMatWrapper(cvImage, 0, &input);

        int32_t width, height;
        vpiImageGetSize(input, &width, &height);
    
        VPIImageFormat type;
        vpiImageGetFormat(input, &type);
    
        VPIImage output;
        vpiImageCreate(width, height, type, 0, &output);
        
        VPIWarpMap map;
        memset(&map, 0, sizeof(map));
        map.grid.numHorizRegions  = 1;
        map.grid.numVertRegions   = 1;
        map.grid.regionWidth[0]   = width;
        map.grid.regionHeight[0]  = height;
        map.grid.horizInterval[0] = 1;
        map.grid.vertInterval[0]  = 1;
        vpiWarpMapAllocData(&map);
        
        VPIFisheyeLensDistortionModel fisheye;
        memset(&fisheye, 0, sizeof(fisheye));
        fisheye.mapping = VPI_FISHEYE_EQUIDISTANT;
        fisheye.k1      = -0.126;
        fisheye.k2      = 0.004;
        fisheye.k3      = 0;
        fisheye.k4      = 0;

        
        float sensorWidth = 22.2; /* APS-C sensor */
        float focalLength = 7.5;
        float f = focalLength*width/sensorWidth;
        const VPICameraIntrinsic K =
        {
            { f, 0, width/2.0 },
            { 0, f, height/2.0 }
        };
        const VPICameraExtrinsic X =
        {
            { 1, 0, 0, 0 },
            { 0, 1, 0, 0 },
            { 0, 0, 1, 0 }
        };

        vpiWarpMapGenerateFromFisheyeLensDistortionModel(K, X, K, &fisheye, &map);
        VPIPayload warp;
        vpiCreateRemap(VPI_BACKEND_CUDA, &map, &warp);

        VPIStream stream;
        vpiStreamCreate(0, &stream);

        vpiSubmitRemap(stream, VPI_BACKEND_CUDA, warp, input, output, VPI_INTERP_CATMULL_ROM, VPI_BORDER_ZERO, 0);

        CHECK_STATUS(vpiStreamSync(stream));

        {
            VPIImageData data;
            CHECK_STATUS(vpiImageLock(output, VPI_LOCK_READ, &data));

            VPIImageData outData;
            CHECK_STATUS(vpiImageLock(output, VPI_LOCK_READ, &outData));

            cv::Mat cvOut(outData.planes[0].height, outData.planes[0].width, CV_8UC3, outData.planes[0].data,
                          outData.planes[0].pitchBytes);

            cv::imwrite("output3.jpg", cvOut);
            
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    if (stream != NULL)
    {
        vpiStreamSync(stream);
    }

    vpiStreamDestroy(stream);
    vpiPayloadDestroy(warp);
    vpiWarpMapFreeData(&map);
    vpiImageDestroy(input);
    vpiImageDestroy(output);

    return retval;
}
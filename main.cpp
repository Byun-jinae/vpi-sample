/*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>

#include <cstring> // for memset
#include <iostream>
#include <sstream>

#include <vpi/WarpMap.h>
#include <vpi/algo/Remap.h>

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
    VPIImage image      = NULL;
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

        vpiImageCreateOpenCVMatWrapper(cvImage, 0, &image);

        vpiImageGetFormat(image,&type);
        vpiImageGetSize(image,&w, &h);
        

        CHECK_STATUS(vpiImageCreate(w, h, type, 0, &output));
        std::cout << w;
        std::cout << h;
        
        vpiStreamCreate(0,&stream);
        memset(&map, 0, sizeof(map));

        map.grid.numHorizRegions  = 1;
        map.grid.numVertRegions   = 1;
        map.grid.regionWidth[0]   = w;
        map.grid.regionHeight[0]  = h;
        map.grid.horizInterval[0] = 1;
        map.grid.vertInterval[0]  = 1;
        vpiWarpMapAllocData(&map);
        vpiWarpMapGenerateIdentity(&map);
        
        int i;
        for (i = 0; i < map.numVertPoints; ++i)
        {
        VPIKeypoint *row = (VPIKeypoint *)((uint8_t *)map.keypoints + map.pitchBytes * i);
        int j;
        for (j = 0; j < map.numHorizPoints; ++j)
        {
        float x = row[j].x - w / 2.0f;
        float y = row[j].y - h / 2.0f;

        const float R = h / 8.0f; /* planet radius */

        const float r = sqrtf(x * x + y * y);

        float theta = M_PI + atan2f(y, x);
        float phi   = M_PI / 2 - 2 * atan2f(r, 2 * R);

        row[j].x = fmod((theta + M_PI) / (2 * M_PI) * (w - 1), w - 1);
        row[j].y = (phi + M_PI / 2) / M_PI * (h - 1);
        }
        }
        vpiCreateRemap(VPI_BACKEND_CUDA, &map, &warp);
        vpiSubmitRemap(stream, VPI_BACKEND_CUDA, warp, image, output, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);

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

    vpiStreamSync(stream);
    
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(warp);
    vpiWarpMapFreeData(&map);
    vpiImageDestroy(image);
    vpiImageDestroy(output);
    return retval;
}

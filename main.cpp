
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// ncnn
#include "gpu.h"
#include "net.h"


static int styletransfer(const ncnn::Net& net, const cv::Mat& bgr, cv::Mat& outbgr)
{
    const int w = bgr.cols;
    const int h = bgr.rows;

    int target_w = 512;
    int target_h = 512;

	const float mean_vals[3] = { 127.5f, 127.5f,  127.5f };
	const float norm_vals[3] = { 1 / 127.5f, 1 / 127.5f, 1 / 127.5f };
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, target_w, target_h);
	in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();

        ex.input("input", in);
        ex.extract("out", out);
    }

    cv::Mat result(out.h, out.w, CV_32FC3);
	for (int i = 0; i < out.c; i++)
	{
		float* out_data = out.channel(i);
		for (int h = 0; h < out.h; h++)
		{
			for (int w = 0; w < out.w; w++)
			{
				result.at<cv::Vec3f>(h, w)[2-i] = out_data[h * out.h + w];
			}
		}
	}
	cv::Mat result8U(out.h, out.w, CV_8UC3);
	result.convertTo(result8U, CV_8UC3, 127.5, 127.5);
    result8U.copyTo(outbgr);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat bgr = cv::imread(imagepath, 1);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    ncnn::create_gpu_instance();

    {
        ncnn::Option opt;
        opt.use_vulkan_compute = true;

        ncnn::Net styletransfernet[3];

        // load
        const char* model_paths[4] = {"celeba.bin", "face_paint_512_v1.bin", "face_paint_512_v2.bin"};
        const char* param_paths[4] = {"celeba.param", "face_paint_512_v1.param", "face_paint_512_v2.param"};
        for (int i = 0; i < 3; i++)
        {
            styletransfernet[i].opt = opt;

            int ret0 = styletransfernet[i].load_param(param_paths[i]);
            int ret1 = styletransfernet[i].load_model(model_paths[i]);

            fprintf(stderr, "load %d %d\n", ret0, ret1);
        }

        // process and save
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < 3; i++)
        {
            cv::Mat outbgr;
            styletransfer(styletransfernet[i], bgr, outbgr);

            char outpath[256];
            sprintf(outpath, "%s.%d.jpg", imagepath, i);
            cv::imwrite(outpath, outbgr);
        }
    }

    ncnn::destroy_gpu_instance();

    return 0;
}

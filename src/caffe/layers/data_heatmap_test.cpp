// Copyright 2015 Tomas Pfister

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/roomnet_test_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <stdint.h>

#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/data_heatmap_test.hpp"
#include "caffe/util/benchmark.hpp"
#include <unistd.h>


namespace caffe
{

template <typename Dtype>
DataHeatmapTestLayer<Dtype>::~DataHeatmapTestLayer<Dtype>() {
    this->StopInternalThread();
}


template<typename Dtype>
void DataHeatmapTestLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    HeatmapTestParameter heatmap_test_param = this->layer_param_.heatmap_test_param();

    // Shortcuts
    const int batchsize = heatmap_data_param.batchsize();
    const int outsize = heatmap_data_param.outsize();
    root_img_dir_ = heatmap_data_param.root_img_dir();

    // load GT
    std::string gt_path = heatmap_data_param.source();
    LOG(INFO) << "Loading annotation from " << gt_path;

    std::ifstream infile(gt_path.c_str());
    string img_name;
    
    // sequential sampling
    while (infile >> img_name)
    {
        img_list_.push_back(img_name);
    }

    // initialise image counter to 0
    cur_img_ = 0;

    // assume input images are RGB (3 channels)
    this->datum_channels_ = 3;

    // init data
    // this->transformed_data_.Reshape(batchsize, this->datum_channels_, outsize, outsize);
    top[0]->Reshape(batchsize, this->datum_channels_, outsize, outsize);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
        this->prefetch_[i].data_.Reshape(batchsize, this->datum_channels_, outsize, outsize);
    this->datum_size_ = this->datum_channels_ * outsize * outsize;

    LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

}




template<typename Dtype>
void DataHeatmapTestLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

    CPUTimer batch_timer;
    batch_timer.Start();
    CHECK(batch->data_.count());
    HeatmapTestParameter heatmap_test_param = this->layer_param_.heatmap_test_param();

    // Pointers to blobs' float data
    Dtype* top_data = batch->data_.mutable_cpu_data();

    cv::Mat img, img_res, img_annotation_vis, img_mean_vis, img_vis, img_res_vis, mean_img_this, seg, segTmp, img_flip;

    // Shortcuts to params
    const int batchsize = heatmap_test_param.batchsize();
    const int outsize = heatmap_test_param.outsize();

    // Shortcuts to global vars
    const int channels = this->datum_channels_;

    // collect "batchsize" images
    std::string img_name;

    // loop over non-augmented images
    for (int idx_img = 0; idx_img < batchsize; idx_img++)
    {
        // get image information
        this->GetCurImg(img_name);

        std::string img_path = this->root_img_dir_ + img_name;
        DLOG(INFO) << "img: " << img_path;
        img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        int width = img.cols;
        int height = img.rows;
        int x_border = width - outsize;
        int y_border = height - outsize;

        // convert from BGR to RGB
        cv::cvtColor(img, img, CV_BGR2RGB);

        // to float
        img.convertTo(img, CV_32FC3);

        if (x_border < 0)
        {
            DLOG(INFO) << "padding " << img_path << " -- not wide enough.";

            cv::copyMakeBorder(img, img, 0, 0, 0, -x_border, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            width = img.cols;
            x_border = width - outsize;

            DLOG(INFO) << "new width: " << width << "   x_border: " << x_border;
        }

        if (y_border < 0)
        {
            DLOG(INFO) << "padding " << img_path << " -- not high enough.";

            cv::copyMakeBorder(img, img, 0, -y_border, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            height = img.rows;
            y_border = height - outsize;

            DLOG(INFO) << "new height: " << height << "   y_border: " << y_border;
        }        

        DLOG(INFO) << "Resizing output image.";

        // resize to output image size
        float resizeFact_x = (float)outsize / (float)img.cols;
        float resizeFact_y = (float)outsize / (float)img.rows;
        DLOG(INFO) << "resizeFact_x: " << resizeFact_x; 
        DLOG(INFO) << "resizeFact_y: " << resizeFact_y; 

        cv::Size s(outsize, outsize);
        cv::resize(img, img_res, s);

        cv::imwrite("test/resized.png", img_res);                

        // resulting image dims
        const int channel_size = outsize * outsize;
        const int img_size = channel_size * channels;

        // store image data
        DLOG(INFO) << "storing image";
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < outsize; i++)
            {
                for (int j = 0; j < outsize; j++)
                {
                    top_data[idx_img_aug * img_size + c * channel_size + i * outsize + j] = img_res.at<cv::Vec3f>(i, j)[c];
                }
            }
        }

        DLOG(INFO) << "next image";

        // move to the next image
        this->AdvanceCurImg();

    } // original image loop

    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}



template<typename Dtype>
void DataHeatmapTestLayer<Dtype>::GetCurImg(string& img_name)
{
    img_name = img_list_[cur_img_];
}

template<typename Dtype>
void DataHeatmapTestLayer<Dtype>::AdvanceCurImg()
{
    if (cur_img_ < img_list_.size() - 1)
        cur_img_++;
    else
        cur_img_ = 0;
}


INSTANTIATE_CLASS(DataHeatmapTestLayer);
REGISTER_LAYER_CLASS(DataHeatmapTest);

} // namespace caffe

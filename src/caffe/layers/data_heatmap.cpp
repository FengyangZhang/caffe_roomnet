// Copyright 2015 Tomas Pfister

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/roomnet_data_layer.hpp"
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

#include "caffe/layers/data_heatmap.hpp"
#include "caffe/util/benchmark.hpp"
#include <unistd.h>


namespace caffe
{

template <typename Dtype>
DataHeatmapLayer<Dtype>::~DataHeatmapLayer<Dtype>() {
    this->StopInternalThread();
}


template<typename Dtype>
void DataHeatmapLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Shortcuts
    const int batchsize = heatmap_data_param.batchsize();
    const int label_width = heatmap_data_param.label_width();
    const int label_height = heatmap_data_param.label_height();
    const int outsize = heatmap_data_param.outsize();
    root_img_dir_ = heatmap_data_param.root_img_dir();

    // initialise rng seed
    const unsigned int rng_seed = caffe_rng_rand();
    srand(rng_seed);

    // load GT
    std::string gt_path = heatmap_data_param.source();
    LOG(INFO) << "Loading annotation from " << gt_path;

    std::ifstream infile(gt_path.c_str());
    string img_name, typeStr, labels;
    
	// sequential sampling
    while (infile >> img_name >> typeStr >> labels)
    {
        // read comma-separated list of regression labels
        std::vector <float> label;
        std::istringstream ss(labels);
        int labelCounter = 1;
        while (ss)
        {
            std::string s;
            if (!std::getline(ss, s, ',')) break;
            label.push_back(atof(s.c_str()));
            labelCounter++;
        }
        int type = atoi(typeStr.c_str());
        img_label_list_.push_back(std::make_pair(img_name, std::make_pair(label, type)));
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

    // init label
    int label_num_channels = 48;
    top[1]->Reshape(batchsize, label_num_channels, label_height, label_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
        this->prefetch_[i].label_.Reshape(batchsize, label_num_channels, label_height, label_width);

    // init type
    vector<int> type_shape(1, batchsize);
    top[2]->Reshape(type_shape);
    
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
        this->prefetch_[i].type_.Reshape(type_shape);

    LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
    LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
    LOG(INFO) << "number of label channels: " << label_num_channels;
    LOG(INFO) << "datum channels: " << this->datum_channels_;

}




template<typename Dtype>
void DataHeatmapLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

    CPUTimer batch_timer;
    batch_timer.Start();
    CHECK(batch->data_.count());
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Pointers to blobs' float data
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();
    Dtype* top_type = batch->type_.mutable_cpu_data();

    cv::Mat img, img_res, img_annotation_vis, img_mean_vis, img_vis, img_res_vis, mean_img_this, seg, segTmp;

    // Shortcuts to params
    const bool is_test = this->layer_param_.is_test();
    const int batchsize = heatmap_data_param.batchsize();
    const int label_height = heatmap_data_param.label_height();
    const int label_width = heatmap_data_param.label_width();
    const int outsize = heatmap_data_param.outsize();
    const int num_aug = 1;

    // Shortcuts to global vars
    const int channels = this->datum_channels_;

    // collect "batchsize" images
    std::vector<float> cur_label;
    std::string img_name;
    int cur_type;

    // loop over non-augmented images
    for (int idx_img = 0; idx_img < batchsize; idx_img++)
    {
        // get image information
        this->GetCurImg(img_name, cur_label, cur_type);

        // get number of channels for image label
        int label_num_valid_channels = cur_label.size();
        int label_num_channels = 48;

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

            // FengyangZhang: ??? don't know why add offset here
            // add border offset to joints
            // for (int i = 0; i < label_num_channels; i += 2)
            //     cur_label[i] = cur_label[i] + x_border;

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


        DLOG(INFO) << "Entering jitter loop.";

        // loop over the jittered versions
        for (int idx_aug = 0; idx_aug < num_aug; idx_aug++)
        {
            // augmented image index in the resulting batch
            const int idx_img_aug = idx_img * num_aug + idx_aug;
            std::vector<float> cur_label_aug = cur_label;

            // store image type
            DLOG(INFO) << "storing type:" << cur_type;
            top_type[idx_img_aug] = cur_type;

            // FengyangZhang: do random horizontal flip
            if (rand() % 2)
            {
                // flip
                cv::flip(img, img, 1);

                // "flip" annotation coordinates
                for (int i = 0; i < label_num_valid_channels; i += 2)
                    cur_label_aug[i] = (float)width - cur_label_aug[i];
            }

            DLOG(INFO) << "Resizing output image.";

            // resize to output image size
            float resizeFact_x = (float)outsize / (float)img.cols;
            float resizeFact_y = (float)outsize / (float)img.rows;
			DLOG(INFO) << "resizeFact_x: " << resizeFact_x;	
			DLOG(INFO) << "resizeFact_y: " << resizeFact_y;	

            cv::Size s(outsize, outsize);
            cv::resize(img, img_res, s);

            if(is_test) {
                cv::imwrite("test/resized.png", img_res);                
            }

            // "resize" annotations
            for (int i = 0; i < label_num_valid_channels; i+=2)
                cur_label_aug[i] *= resizeFact_x;

            for (int i = 1; i < label_num_valid_channels; i+=2)
                cur_label_aug[i] *= resizeFact_y;

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

            const int type_ind_range[12] = {0, 8, 14, 20, 24, 28, 34, 38, 42, 44, 46, 48};
            int low_indice = type_ind_range[cur_type];
            int high_indice = type_ind_range[cur_type + 1];
            // store label as gaussian
            DLOG(INFO) << "storing labels";
            const int label_channel_size = label_height * label_width;
            const int label_img_size = label_channel_size * label_num_channels;
            float label_resize_fact = (float) label_height / (float) outsize;
            float sigma = 1.5;

            // set ground truth label on corresponding channels
            for (int idx_ch = low_indice; idx_ch < high_indice; idx_ch++)
            {
                float x = label_resize_fact * cur_label_aug[2*(idx_ch-low_indice)];
                float y = label_resize_fact * cur_label_aug[2*(idx_ch-low_indice)+1];
                // DLOG(INFO) << x << ", " << y;
                
                for (int i = 0; i < label_height; i++)
                {
                    for (int j = 0; j < label_width; j++)
                    {
                        int label_idx = idx_img_aug * label_img_size + idx_ch * label_channel_size + i * label_width + j;
                        float gaussian = ( 1 / ( sigma * sqrt(2 * M_PI) ) ) * exp( -0.5 * ( pow(i - y, 2.0) + pow(j - x, 2.0) ) * pow(1 / sigma, 2.0) );
                        gaussian = 4 * gaussian;
                        top_label[label_idx] = gaussian;
                    }
                }
            }

            // set ground truth value on unrelated channels to zero
            for (int idx_ch = 0; idx_ch < low_indice; idx_ch++)
            {
                for (int i = 0; i < label_height; i++)
                {
                    for (int j = 0; j < label_width; j++)
                    {
                        int label_idx = idx_img_aug * label_img_size + idx_ch * label_channel_size + i * label_height + j;
                        top_label[label_idx] = 0;
                    }
                }
            }
            for (int idx_ch = high_indice; idx_ch < 48; idx_ch++)
            {
                for (int i = 0; i < label_height; i++)
                {
                    for (int j = 0; j < label_width; j++)
                    {
                        int label_idx = idx_img_aug * label_img_size + idx_ch * label_channel_size + i * label_height + j;
                        top_label[label_idx] = 0;
                    }
                }
            }

        } // jittered versions loop

        DLOG(INFO) << "next image";

        // move to the next image
        this->AdvanceCurImg();

    } // original image loop

    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}



template<typename Dtype>
void DataHeatmapLayer<Dtype>::GetCurImg(string& img_name, std::vector<float>& img_label, int& img_type)
{

    img_name = img_label_list_[cur_img_].first;
    img_label = img_label_list_[cur_img_].second.first;
    img_type = img_label_list_[cur_img_].second.second;
}

template<typename Dtype>
void DataHeatmapLayer<Dtype>::AdvanceCurImg()
{
    if (cur_img_ < img_label_list_.size() - 1)
        cur_img_++;
    else
        cur_img_ = 0;
}

template<typename Dtype>
void DataHeatmapLayer<Dtype>::VisualiseAnnotations(cv::Mat img_annotation_vis, int label_num_channels, std::vector<float>& img_class, int multfact)
{
    // colors
    const static cv::Scalar colors[] = {
        CV_RGB(0, 0, 255),
        CV_RGB(0, 128, 255),
        CV_RGB(0, 255, 255),
        CV_RGB(0, 255, 0),
        CV_RGB(255, 128, 0),
        CV_RGB(255, 255, 0),
        CV_RGB(255, 0, 0),
        CV_RGB(255, 0, 255)
    };

    int numCoordinates = int(label_num_channels / 2);

    // points
    cv::Point centers[numCoordinates];
    for (int i = 0; i < label_num_channels; i += 2)
    {
        int coordInd = int(i / 2);
        centers[coordInd] = cv::Point(img_class[i] * multfact, img_class[i + 1] * multfact);
        cv::circle(img_annotation_vis, centers[coordInd], 1, colors[coordInd], 3);
    }

    // connecting lines
    cv::line(img_annotation_vis, centers[1], centers[3], CV_RGB(0, 255, 0), 1, CV_AA);
    cv::line(img_annotation_vis, centers[2], centers[4], CV_RGB(255, 255, 0), 1, CV_AA);
    cv::line(img_annotation_vis, centers[3], centers[5], CV_RGB(0, 0, 255), 1, CV_AA);
    cv::line(img_annotation_vis, centers[4], centers[6], CV_RGB(0, 255, 255), 1, CV_AA);
}


template <typename Dtype>
float DataHeatmapLayer<Dtype>::Uniform(const float min, const float max) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = max - min;
    float r = random * diff;
    return min + r;
}

template <typename Dtype>
cv::Mat DataHeatmapLayer<Dtype>::RotateImage(cv::Mat src, float rotation_angle)
{
    cv::Mat rot_mat(2, 3, CV_32FC1);
    cv::Point center = cv::Point(src.cols / 2, src.rows / 2);
    double scale = 1;

    // Get the rotation matrix with the specifications above
    rot_mat = cv::getRotationMatrix2D(center, rotation_angle, scale);

    // Rotate the warped image
    cv::warpAffine(src, src, rot_mat, src.size());

    return rot_mat;
}

INSTANTIATE_CLASS(DataHeatmapLayer);
REGISTER_LAYER_CLASS(DataHeatmap);

} // namespace caffe

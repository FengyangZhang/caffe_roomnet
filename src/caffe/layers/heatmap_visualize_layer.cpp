#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/heatmap_visualize_layer.hpp"


// Euclidean loss layer that computes loss on a [x] x [y] x [ch] set of heatmaps,
// and enables visualisation of inputs, GT, prediction and loss.


namespace caffe {

template <typename Dtype>
void HeatmapVisualizeLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    //LossLayer<Dtype>::Reshape(bottom, top);
}


template<typename Dtype>
void HeatmapVisualizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void HeatmapVisualizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_pred = bottom[0]->cpu_data(); // predictions for all images

    const int num_images = bottom[0]->num();
    const int label_height = bottom[0]->height();
    const int label_width = bottom[0]->width();
    const int num_channels = bottom[0]->channels();

    DLOG(INFO) << "bottom size: " << bottom[0]->height() << " " << bottom[0]->width() << " " << bottom[0]->channels();

    const int label_channel_size = label_height * label_width;
    const int label_img_size = label_channel_size * num_channels;
    cv::Mat bottom_img, bottom_img_8bit;  // Initialise opencv images for visualisation
    double min;
    double max;
    int img_id = 0;
    bottom_img = cv::Mat::zeros(label_height, label_width, CV_32FC1);
    bottom_img_8bit = cv::Mat::zeros(label_height, label_width, CV_8UC1);

    // Loop over images
    for (int idx_img = 0; idx_img < num_images; idx_img++)
    {
        for (int idx_ch = 0; idx_ch < num_channels; idx_ch++)
        {
            for (int i = 0; i < label_height; i++)
            {
                for (int j = 0; j < label_width; j++)
                {
                    int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_width + j;
                    // Store visualisation for each channel
                    bottom_img.at<float>((int)i, (int)j) = (float)(bottom_pred[image_idx]);
                }
            }

            std::stringstream ss;
            ss << img_id << "-" << idx_ch;
            string str = ss.str();
            cv::minMaxIdx(bottom_img, &min, &max);
            cv::convertScaleAbs(bottom_img, bottom_img_8bit, 255 / max);
            cv::imwrite("test/" + str + ".png", bottom_img_8bit);
            
        }
        img_id++;
    }
}


template <typename Dtype>
void HeatmapVisualizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
}


template <typename Dtype>
void HeatmapVisualizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}

template <typename Dtype>
void HeatmapVisualizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}



#ifdef CPU_ONLY
STUB_GPU(HeatmapVisualizeLayer);
#endif

INSTANTIATE_CLASS(HeatmapVisualizeLayer);
REGISTER_LAYER_CLASS(HeatmapVisualize);


}  // namespace caffe

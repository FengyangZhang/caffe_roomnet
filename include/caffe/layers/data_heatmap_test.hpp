// Copyright 2014 Tomas Pfister

#ifndef CAFFE_HEATMAP_TEST_HPP_
#define CAFFE_HEATMAP_TEST_HPP_

#include "caffe/layer.hpp"
#include <vector>
#include <boost/timer/timer.hpp>
#include <opencv2/core/core.hpp>

#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{


template<typename Dtype>
class DataHeatmapTestLayer: public RoomnetPrefetchingTestLayer<Dtype>
{

public:

    explicit DataHeatmapTestLayer(const LayerParameter& param)
        : RoomnetPrefetchingTestLayer<Dtype>(param) {}
    virtual ~DataHeatmapTestLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "DataHeatmapTest"; }

    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 1; }


protected:
    virtual void load_batch(Batch<Dtype>* batch);

    // Filename of current image
    inline void GetCurImg(string& img_name);

    inline void AdvanceCurImg();

    // Global vars
    int datum_channels_;
    string root_img_dir_;
    int cur_img_; // current image index

    // vector of (image) pairs
    vector< string > img_list_;
};

}

#endif /* CAFFE_HEATMAP_TEST_HPP_ */

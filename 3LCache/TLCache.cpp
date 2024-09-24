#include "TLCache.h"
#include <algorithm>
#include "utils.h"
#include <chrono>


using namespace chrono;
using namespace std;
using namespace TLCache;


void TLCacheCache::train() {
    ++n_retrain;
    auto timeBegin = chrono::system_clock::now();
    if (booster) LGBM_BoosterFree(booster);
    // create training dataset
    DatasetHandle trainData;
    LGBM_DatasetCreateFromCSR(
            static_cast<void *>(training_data->indptr.data()),
            C_API_DTYPE_INT32,
            training_data->indices.data(),
            static_cast<void *>(training_data->data.data()),
            C_API_DTYPE_FLOAT64,
            training_data->indptr.size(),
            training_data->data.size(),
            n_feature,  //remove future t
            training_params,
            nullptr,
            &trainData);

    LGBM_DatasetSetField(trainData,
                         "label",
                         static_cast<void *>(training_data->labels.data()),
                         training_data->labels.size(),
                         C_API_DTYPE_FLOAT32);

    // init booster
    LGBM_BoosterCreate(trainData, training_params, &booster);
    // train
    for (int i = 0; i < stoi(training_params["num_iterations"]); i++) {
        int isFinished;
        LGBM_BoosterUpdateOneIter(booster, &isFinished);
        if (isFinished) {
            break;
        }
    }

    LGBM_DatasetFree(trainData);
    training_time += chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - timeBegin).count();
    pred_map.clear();
    pred_times.clear();
    pred_times.shrink_to_fit();

    MAX_EVICTION_BOUNDARY[0] = MAX_EVICTION_BOUNDARY[1];

    origin_current_seq = current_seq;
    if (n_req > pow(10, 6) && is_full) {
        if ((n_window_hit - n_hit) * 1.0 / (n_hit * hsw) > 0.01) { 
            if (hsw < (n_req - n_hit) * 1.0 / (n_window_hit - n_hit) - 1)
                hsw += 1, is_full = 0;
            hsw = fmin(hsw, 10.0);
        }    
        n_hit = 0, n_window_hit = 0, n_req = 0;
    }
}


void TLCacheCache::sample() {
    auto rand_idx = _distribution(_generator);
    uint32_t pos = rand_idx % (in_cache.metas.size() + out_cache.metas.size());
    auto &meta = pos < in_cache.metas.size() ? in_cache.metas[pos] : out_cache.metas[pos - in_cache.metas.size()];
    meta.emplace_sample(current_seq);
}


void TLCacheCache::update_stat_periodic() {
    float percent_beyond;
    segment_percent_beyond.emplace_back(percent_beyond);
    segment_n_retrain.emplace_back(n_retrain);

    float positive_example_ratio;
    if (0 == training_data_distribution[0] && 0 == training_data_distribution[1]) {
        positive_example_ratio = 0;
    } else {
        positive_example_ratio = static_cast<float>(training_data_distribution[1])/(training_data_distribution[0] + training_data_distribution[1]);
    }
    training_data_distribution[0] = training_data_distribution[1] = 0;
    segment_positive_example_ratio.emplace_back(positive_example_ratio);

    n_retrain = 0;
}

// bool TLCacheCache::compareHeapUint(const HeapUint& a, const HeapUint& b) {
//     return a.reuse_time < b.reuse_time; // 按照 reuse_time 降序排序
// }


bool TLCacheCache::lookup(const SimpleRequest &req) {
    bool ret;
    ++current_seq;
    if (is_full == 1)
        n_req++;
    auto it = key_map.find(req.id);
    if (it != key_map.end()) {
        auto list_idx = it->second.list_idx;
        auto list_pos = it->second.list_pos;
        if (is_full == 1) {
            if (list_idx == 0)
                n_hit++;
            n_window_hit++;
        }
        // 找到对应的窗口内的对象请求
        Meta &meta = list_idx == 0 ?  in_cache.metas[list_pos]: out_cache.metas[uint32_t(list_pos - out_cache.front_index)];
        auto sample_time = meta._sample_times;
        if (sample_time != 0 && (_distribution(_generator) % 4 == 0 || !booster)) {
        // if (sample_time != 0 && (_distribution(_generator) % 2 == 0 || !booster)) {
            uint32_t future_distance = current_seq - sample_time;
            training_data->emplace_back(meta, sample_time, future_distance, meta._key);
            //batch_size ~>= batch_size
            if (training_data->labels.size() >= batch_size && evict_nums <= 0) {
                train();
                training_data->clear();
            }
            meta._sample_times = 0;
        } else {
            meta._sample_times = 0;
        }
        // 如果对象位于out_cache中
        // 命中时
        meta.update(current_seq);
        if (!list_idx) { 
            // 基于lru进行采样
            // 如果当前指针命中，则将指针移动到下一个位置 
            if(samplepointer == list_pos){
                samplepointer = in_cache.dq[samplepointer].next;
            }
            if(pred_map.find(req.id) != pred_map.end()){
                pred_map.erase(req.id);
            }
            in_cache.re_request(list_pos);
        }
        ret = !list_idx;
    } else {
        // 如果对象没有在记忆窗口出现过   
        ret = false;
    }
    if (is_sampling) {
        sample();
    }

    erase_out_cache();
    return ret;
}

void TLCacheCache::erase_out_cache() {
    if (out_cache.metas.size() >= max_out_cache_size) {
        if (is_full == 0)
            is_full = 1;
        auto &meta = out_cache.metas[0];
        if (meta._size != 0) {
            auto sample_time = meta._sample_times;
            // uint32_t frac = one_hit_wonder > 0.5 * batch_size ? 50: 2;
            // if (sample_time != 0 && (_distribution(_generator) % frac == 0 || !booster)) {
            if (sample_time != 0 && (_distribution(_generator) % 32 == 0 || !booster)) {
                uint32_t future_distance = MAX_EVICTION_BOUNDARY[0] + current_seq - meta._past_timestamp;
                if (MAX_EVICTION_BOUNDARY[1] < current_seq - meta._past_timestamp)
                    MAX_EVICTION_BOUNDARY[1] = current_seq - meta._past_timestamp;
                training_data->emplace_back(meta, sample_time, future_distance, meta._key);
                if (training_data->labels.size() >= batch_size && evict_nums <= 0) {
                    train();
                    training_data->clear();
                }
            }
            key_map.erase(meta._key);
            meta.free();
        }
        out_cache.metas.pop_front();
        out_cache.front_index++;
    }
}

void TLCacheCache::admit(const SimpleRequest &req) {
    const uint64_t &size = req.size;
    if (size > _cacheSize) {
        LOG("L", _cacheSize, req.id, size);
        return;
    }
    max_out_cache_size = in_cache.metas.size() * hsw + 2;
    // 未开始替换对象时，我们只需要在out_cache中更新数据即可
    auto it = key_map.find(req.id);
    uint32_t pos;
    if (it == key_map.end()){
        // 当缓存的对象信息数超过阈值，则驱逐最久未访问的对象，此处可以考虑训练数据的采样
        pos = in_cache.metas.size();
        in_cache.metas.emplace_back(Meta(req.id, req.size, current_seq, req.extra_features));
        in_cache.dq.emplace_back(CircleList());
    } else {
        pos = in_cache.metas.size();
        in_cache.metas.emplace_back(out_cache.metas[uint32_t(it->second.list_pos - out_cache.front_index)]);
        out_cache.metas[uint32_t(it->second.list_pos - out_cache.front_index)]._size = 0;
        in_cache.dq.emplace_back(CircleList());
    }  
    in_cache.request(pos);
    key_map[req.id] = {0, pos};
    _currentSize += size;
    // 记录新对象
    if (booster) {
        new_obj_size += req.size;
        new_obj_keys.emplace_back(req.id);
    }
    // while (_currentSize > _cacheSize) { 
    //     evict();  
    // }
}

uint32_t TLCacheCache::rank() {
// 新对象的采样
    vector<uint32_t> sampled_objects;
    sampled_objects = quick_demotion();
    unsigned int idx_row = 0;
    // idx_row = 0;
    uint16_t count = 0;
    if (initial_queue_length == 0) {
        initial_queue_length = in_cache.metas.size();
    }
    if (sample_rate >= initial_queue_length * reserved_space * 1.0 / 100 + eviction_rate)
        sample_rate = initial_queue_length * reserved_space * 1.0 / 100 + eviction_rate;
    uint16_t freq = 0;
    while (idx_row < sample_rate) {
        freq = in_cache.metas[samplepointer]._freq - 1;
        if (evcition_distribution[3] == 0 && scan_length > initial_queue_length * sampling_lru * 1.0 / 100){
            evcition_distribution[2] = evcition_distribution[0], evcition_distribution[3] = evcition_distribution[1];
            evcition_distribution[1] = 0, evcition_distribution[0] = 0;
        }
        if (freq  < sample_boundary || scan_length <= initial_queue_length * sampling_lru * 1.0 / 100 + eviction_rate) {
            sampled_objects.emplace_back(samplepointer);
            idx_row++;
        }

        scan_length++;
        
        if (scan_length >= initial_queue_length) {
            
            // 对象丢失率不需要进行调整，顺序扫描就行了，只需要将sample_boundary调整到最大即可
            if (objective == object_miss_ratio) {
                initial_queue_length = in_cache.metas.size();
                samplepointer = in_cache.q.head;
                scan_length = 0;
                pred_map.clear();
                pred_times.clear();
                pred_times.shrink_to_fit();
                continue;
            }
            uint32_t eviciton_sum = 0, p99 = 0;
            for (int i =  0; i < 16; i++)
                eviciton_sum += object_distribution_n_eviction[i];
            for (int i = 0; i < 16; i++) {  
                // 选择99分位的请求次数
                p99 += object_distribution_n_eviction[i];
                if (p99 >= 0.99 * eviciton_sum) {
                    if (i == 0)
                        sample_boundary = 1;
                    else                
                        sample_boundary = pow(2, i - 1) + ceil((pow(2, i) - pow(2, i - 1)) * ((0.99 * eviciton_sum + object_distribution_n_eviction[i] - p99) / object_distribution_n_eviction[i]));
                    break;
                }
            }
            if (evcition_distribution[2] * evcition_distribution[1] > evcition_distribution[0] * evcition_distribution[3])
                sampling_lru++;
            else if (sampling_lru > 1)
                sampling_lru--;
            if ((evcition_distribution[0] + evcition_distribution[2]) > new_obj_keys.size() * (reserved_space + 1) * 1.0 / reserved_space)
                reserved_space++;
            else if (reserved_space > 1 && (evcition_distribution[0] + evcition_distribution[2]) < new_obj_keys.size())
                reserved_space /= 2;
            memset(evcition_distribution, 0, sizeof(uint64_t) * 4);
            memset(object_distribution_n_eviction, 0, sizeof(uint32_t) * 16);
            initial_queue_length = in_cache.metas.size();
            samplepointer = in_cache.q.head;
            scan_length = 0;
            pred_map.clear();
            pred_times.clear();
            pred_times.shrink_to_fit();
            continue;
        }
        samplepointer = in_cache.dq[samplepointer].next;
    }
    prediction(sampled_objects);
    return sampled_objects.size();
}

vector<uint32_t> TLCacheCache::quick_demotion() {
    vector<uint32_t> sampled_objects;
    int i, j = 0;
    while (new_obj_size > _currentSize * reserved_space * 1.0 / 100  && j < sample_rate * 2 && i < new_obj_keys.size()) {
        auto it = key_map[new_obj_keys[i]];
        if (it.list_idx == 0) {
            new_obj_size -= in_cache.metas[it.list_pos]._size;
            sampled_objects.emplace_back(it.list_pos);
            j++;
        } else {
            new_obj_size -= out_cache.metas[it.list_pos - out_cache.front_index]._size;
        }
        i++;
    }
    new_obj_keys.erase(new_obj_keys.begin(), new_obj_keys.begin() + i);
    if (new_obj_keys.size() == 0)
        new_obj_size = 0;
    return sampled_objects;
}

// void TLCacheCache::evict() {
//     is_sampling = true;
//     auto epair = rank();
//     evict_with_candidate(epair);
// }

void TLCacheCache::evict() {
    auto epair = evict_predobj();
    evict_with_candidate(epair);
}

void TLCacheCache::evict_with_candidate(pair<uint64_t, uint32_t> &epair) {
    is_sampling = true;
    evict_nums -= 1;
    uint64_t key = epair.first;
    uint32_t old_pos = epair.second;
    _currentSize -= in_cache.metas[old_pos]._size;

    pred_map.erase(key);
    //must be the tail of lru
    if (old_pos == samplepointer){
        samplepointer = in_cache.dq[samplepointer].next;
    }
    key_map[key] = {1, uint32_t(out_cache.metas.size()) + out_cache.front_index};
    out_cache.metas.emplace_back(in_cache.metas[old_pos]);
    in_cache.erase(old_pos);
    uint32_t in_cache_tail_idx = in_cache.metas.size() - 1;
    // 调整链表
    if (old_pos != in_cache_tail_idx) {
        if (samplepointer == in_cache_tail_idx){
            samplepointer = in_cache.dq[in_cache_tail_idx].next;
        }
        key_map[in_cache.metas.back()._key].list_pos = old_pos;
        in_cache.metas[old_pos] = in_cache.metas.back();
        in_cache.dq[old_pos] = in_cache.dq[in_cache_tail_idx];
        in_cache.dq[in_cache.dq[old_pos].prev].next = old_pos;
        in_cache.dq[in_cache.dq[old_pos].next].prev = old_pos;
        if (in_cache.q.tail == in_cache_tail_idx)
            in_cache.q.tail = old_pos;
        else if (in_cache.q.head == in_cache_tail_idx)
            in_cache.q.head = old_pos;
    }
    
    in_cache.metas.pop_back();
    in_cache.dq.pop_back();
}

// 驱逐已预测的对象
pair<uint64_t, uint32_t> TLCacheCache::evict_predobj(){
    {
        //if not trained yet, or in_cache_lru past memory window, use LRU
        auto pos = in_cache.q.head;
        auto &meta = in_cache.metas[pos];
        if (!booster) {
            return {meta._key, pos};
        } 
    }
    // 使用堆排序
    // 通过一个map和堆进行优化，map用于处理缓存状态更新的问题，堆用于处理驱逐时的问题
    // map防止对象更新状态后重新采样，而导致报错
    if (evict_nums <= 0 || pred_map.empty()) {
        evict_nums = rank() / eviction_rate;
    }
    
    float reuse_time;
    uint64_t key;
    while (!pred_times.empty())
    {
        reuse_time = pred_times.front().reuse_time;
        key = pred_times.front().key;
        // pop_heap(pred_times.begin(), pred_times.end(), compareHeapUint);
        pop_heap(pred_times.begin(), pred_times.end(), 
        [](const HeapUint& a, const HeapUint& b) {
            return a.reuse_time < b.reuse_time;
        });
        pred_times.pop_back();
        if(pred_map.find(key) != pred_map.end() && pred_map[key] == reuse_time){
            uint32_t old_pos = key_map[key].list_pos;
            object_distribution_n_eviction[uint16_t(log2(in_cache.metas[old_pos]._freq))]++;
            // 区分新对象和旧对象的驱逐比例，0表示新对象
            if (in_cache.metas[old_pos]._past_timestamp < in_cache.metas[samplepointer]._past_timestamp) 
                evcition_distribution[0]++;
            evcition_distribution[1]++;
            return {key, old_pos};
        }
    }

    return {-1, -1};
}

void TLCacheCache::prediction(vector<uint32_t> sampled_objects) {
    // auto timeBegin = chrono::system_clock::now();
    uint32_t sample_nums = sampled_objects.size();
    int32_t indptr[sample_nums + 1];
    indptr[0] = 0;
    int32_t indices[sample_nums * n_feature];
    double data[sample_nums * n_feature];
    int32_t past_timestamps[sample_nums];
    uint32_t sizes[sample_nums];

    // unordered_set<uint64_t> key_set;
    uint64_t keys[sample_nums];
    uint32_t poses[sample_nums];
    unsigned int idx_feature = 0;
    uint32_t pos;
    unsigned int idx_row = 0;
    for (; idx_row < sample_nums; idx_row++) {
        // 使用lru的方法进行采样驱逐，修改点
        pos = sampled_objects[idx_row];
        auto &meta = in_cache.metas[pos];
        keys[idx_row] = meta._key;
        poses[idx_row] = pos;
        indices[idx_feature] = 0;
        // 年龄
        data[idx_feature++] = current_seq - meta._past_timestamp;
        
        past_timestamps[idx_row] = meta._past_timestamp;

        uint8_t j = 0;
        // uint32_t this_past_distance = 0;
        uint16_t n_within = meta._freq;
        if (meta._extra) {
            for (j = 0; j < meta._extra->_past_distance_idx && j < max_n_past_distances; ++j) {
                uint8_t past_distance_idx = (meta._extra->_past_distance_idx - 1 - j) % max_n_past_distances;
                uint32_t &past_distance = meta._extra->_past_distances[past_distance_idx];
                // this_past_distance += past_distance;
                indices[idx_feature] = j + 1;
                data[idx_feature++] = past_distance;
            }
            // n_within = meta._extra->_past_distances.size();
        }

        indices[idx_feature] = max_n_past_timestamps;
        data[idx_feature++] = meta._size;
        sizes[idx_row] = meta._size;

        indices[idx_feature] = max_n_past_timestamps + 1;
        data[idx_feature++] = n_within;

        //remove future t
        indptr[idx_row + 1] = idx_feature;
    }
    int64_t len;
    double scores[sample_nums];
    LGBM_BoosterPredictForCSR(booster,
                              static_cast<void *>(indptr),
                              C_API_DTYPE_INT32,
                              indices,
                              static_cast<void *>(data),
                              C_API_DTYPE_FLOAT64,
                              idx_row + 1,
                              idx_feature,
                              n_feature,  //remove future t
                              C_API_PREDICT_NORMAL,
                              0,
                              inference_params,
                              &len,
                              scores);
    float _distance;
    if (objective == byte_miss_ratio) {
        for (int i = 0; i < sample_nums; ++i) {
            // _distance = pow(scores[i], 2) + uint64_t(current_seq - origin_current_seq);
            _distance = exp(scores[i]) + uint64_t(current_seq - origin_current_seq);
            pred_times.push_back({_distance, keys[i]});
            push_heap(pred_times.begin(), pred_times.end(), 
            [](const HeapUint& a, const HeapUint& b) {
                return a.reuse_time < b.reuse_time;
            });
            pred_map[keys[i]] = _distance;
        }
    } else {
        for (int i = 0; i < sample_nums; ++i) {
            _distance = float(sizes[i] * exp(scores[i]));
            pred_times.push_back({_distance, keys[i]});
            push_heap(pred_times.begin(), pred_times.end(), 
            [](const HeapUint& a, const HeapUint& b) {
                return a.reuse_time < b.reuse_time;
            });
            pred_map[keys[i]] = _distance;
        }
    }
    // inference_time += chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - timeBegin).count();
}

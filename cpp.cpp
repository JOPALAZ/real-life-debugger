#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

const int INPUT_WIDTH = 320;
const int INPUT_HEIGHT = 320;
const float CONFIDENCE_THRESHOLD = 0.5f;

// Function to preprocess the image
std::vector<float> preprocess(const cv::Mat& frame) {
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

    cv::Mat rgb_frame;
    cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);

    rgb_frame.convertTo(rgb_frame, CV_32F, 1.0 / 255.0);

    // Convert from HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(rgb_frame, channels);
    std::vector<float> input_tensor_values;
    for (int c = 0; c < 3; ++c) {
        input_tensor_values.insert(
            input_tensor_values.end(),
            (float*)channels[c].datastart,
            (float*)channels[c].dataend
        );
    }

    return input_tensor_values;
}

// Function to post-process the model output and draw bounding boxes
void post_process(const float* output_data, const std::vector<int64_t>& output_shape, cv::Mat& frame) {
    int64_t num_boxes = output_shape[1];
    int64_t box_info = output_shape[2];

    if (num_boxes == 0) {
        std::cout << "No objects to display." << std::endl;
        return;
    }

    float x_scale = static_cast<float>(frame.cols) / INPUT_WIDTH;
    float y_scale = static_cast<float>(frame.rows) / INPUT_HEIGHT;

    if (box_info >= 6) {
        for (int64_t i = 0; i < num_boxes; ++i) {
            const float* ptr = output_data + i * box_info;
            float xcenter = ptr[0];
            float ycenter = ptr[1];
            float width = ptr[2];
            float height = ptr[3];
            float conf = ptr[4];
            float cls = ptr[5];

            if (conf > CONFIDENCE_THRESHOLD) {
                int x1 = static_cast<int>((xcenter - width / 2.0f) * x_scale);
                int y1 = static_cast<int>((ycenter - height / 2.0f) * y_scale);
                int x2 = static_cast<int>((xcenter + width / 2.0f) * x_scale);
                int y2 = static_cast<int>((ycenter + height / 2.0f) * y_scale);

                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                std::string label = "Class: " + std::to_string(static_cast<int>(cls)) + ", Conf: " + std::to_string(conf);
                cv::putText(frame, label, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                std::cout << "Box: (" << x1 << ", " << y1 << "), (" << x2 << ", " << y2 << "), Confidence: " << conf << ", Class: " << cls << std::endl;
            }
        }
    } else {
        std::cout << "Invalid output data from the model: ";
        for (auto dim : output_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Object Detection");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    const char* model_path = "model.onnx";

    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, model_path, session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load the model: " << e.what() << std::endl;
        return -1;
    }

    // Get input and output node names
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::AllocatedStringPtr> input_names_ptrs;
    std::vector<const char*> input_node_names;
    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        input_names_ptrs.emplace_back(session.GetInputNameAllocated(i, allocator));
        input_node_names.push_back(input_names_ptrs.back().get());
    }

    std::vector<Ort::AllocatedStringPtr> output_names_ptrs;
    std::vector<const char*> output_node_names;
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        output_names_ptrs.emplace_back(session.GetOutputNameAllocated(i, allocator));
        output_node_names.push_back(output_names_ptrs.back().get());
    }

    // Capture video from webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Failed to open the camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        // Check for 'q' key press to exit
        if (cv::waitKey(1) == 'q') {
            break;
        }

        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture frame." << std::endl;
            break;
        }

        // Preprocess the image
        std::vector<float> input_tensor_values = preprocess(frame);

        // Create input tensor
        std::vector<int64_t> input_dims = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size()
        );

        // Run inference
        std::vector<Ort::Value> output_tensors;
        try {
            output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_node_names.data(), &input_tensor, 1,
                output_node_names.data(), output_node_names.size()
            );
        } catch (const Ort::Exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            continue;
        }

        // Process the output
        auto& output_tensor = output_tensors.front();
        auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        float* output_data = output_tensor.GetTensorMutableData<float>();

        post_process(output_data, output_shape, frame);

        // Display the result
        cv::imshow("Object Detection", frame);
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    // No need to manually free input/output names; they are managed by Ort::AllocatedStringPtr

    return 0;
}

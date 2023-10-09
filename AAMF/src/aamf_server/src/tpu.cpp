#include "tpu.h"

std::unique_ptr interpreter; 

std::unique_ptr init_interpreter(void)
{
    const std::string model_path = "/path/to/model_compiled_for_edgetpu.tflite";
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
        edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    std::unique_ptr<tflite::Interpreter> model_interpreter =
        BuildEdgeTpuInterpreter(*model, edgetpu_context.get());

    std::unique_ptr BuildEdgeTpuInterpreter(
        const tflite::FlatBufferModel &model,
        edgetpu::EdgeTpuContext *edgetpu_context)
    {
        tflite::ops::builtin::BuiltinOpResolver resolver;
        resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
        std::unique_ptr interpreter;
        if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk)
        {
            std::cerr << "Failed to build interpreter." << std::endl;
        }
        // Bind given context with interpreter.
        interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
        interpreter->SetNumThreads(1);
        if (interpreter->AllocateTensors() != kTfLiteOk)
        {
            std::cerr << "Failed to allocate tensors." << std::endl;
        }
        return interpreter;
    }
    return -1;
}
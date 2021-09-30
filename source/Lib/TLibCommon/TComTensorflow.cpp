
#include "TComTensorflow.h"

using namespace tensorflow;
using namespace std;

int NNPredict::a = 0;

tensorflow::Session* NNPredict::session_4[NUM_MODE] = { NULL };
tensorflow::Session* NNPredict::session_8[NUM_MODE] = { NULL };
tensorflow::Session* NNPredict::session_16[NUM_MODE] = { NULL };
tensorflow::Session* NNPredict::session_32[NUM_MODE] = { NULL };

int NNPredict::load_pb()
{

		// Read in the protobuf graph we exported
		// (The path seems to be relative to the cwd. Keep this in mind
		// when using `bazel run` since the cwd isn't where you call
		// `bazel run` but from inside a temp folder.)

#if LINUX
string BaseAdd = "/home/sun/hevc_cnn/models/";
#else
string BaseAdd = "D:/hevc_cnn/models/";

#endif
	
#if LIGHT_MODEL
		string pb_4x4 = BaseAdd + "fc/128dim_4layer/p_prune_4_tip_4x512_xavier_";
		//string pb_4x4 = BaseAdd + "fc/64dim_4layer/4_tip_4x64_xavier_";

		string pb_8x8 = BaseAdd + "fc/128dim_4layer/8_tip_4x128_xavier_";
#if inria_tu16
		string pb_16x16 = BaseAdd + "conv/16dim_4layer/16_inria_filter16_xavier_";
#else
		string pb_16x16 = BaseAdd + "conv/16dim_4layer/16_inria_filter16_xavier_";
#endif

		string pb_32x32 = BaseAdd + "conv/16dim_4layer/32_inria_filter16_xavier_";

#else
		string pb_4x4 = BaseAdd + "fc/512dim_4layer/4_tip_4x512_xavier_";
		string pb_8x8 = BaseAdd + "fc/1024dim_4layer/8_tip_4x1024_xavier_";

		string pb_16x16 = BaseAdd + "conv/64dim_4layer/16_inria_filter64_xavier_";
		string pb_32x32 = BaseAdd + "conv/64dim_4layer/32_inria_filter64_v2_";
#endif

		for (int i = 0; i < NUM_MODE; i++)
		{
			string idx = to_string(i);
			load_pb_kernel(pb_4x4 + idx + ".pb", graph_def_4[i]);
			load_pb_kernel(pb_8x8 + idx + ".pb", graph_def_8[i]);
			load_pb_kernel(pb_16x16 + idx + ".pb", graph_def_16[i]);
			load_pb_kernel(pb_32x32 + idx + ".pb", graph_def_32[i]);
		}

		return 0;

}


int NNPredict::load_pb_kernel(std::string model_add, tensorflow::GraphDef &graph_def)
{
	using namespace tensorflow;

	Status status = ReadBinaryProto(Env::Default(), model_add, &graph_def);
	if (!status.ok()) {
		//std::cout << status.ToString() << "\n";
		return 1;
	}

	return 0;

}

int NNPredict::new_session_kernel(tensorflow::Session* &session, tensorflow::GraphDef &graph_def)
{
	using namespace tensorflow;

	SessionOptions opts;
	

	//opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
	//opts.config.mutable_gpu_options()->set_visible_device_list("0");

	//graph::SetDefaultDevice("/cpu:0", &graph_def);
	// new 32x32
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		//std::cout << status.ToString() << "\n";
		return 1;
	}
	// Add the graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) {
		//std::cout << status.ToString() << "\n";
		return 1;
	}

	

	return 0;

}

int NNPredict::new_session()
{
	// Initialize a tensorflow session
	using namespace tensorflow;

	load_pb();


	for (int i = 0; i < NUM_MODE; i++)
	{
		new_session_kernel(session_4[i], graph_def_4[i]);
		new_session_kernel(session_8[i], graph_def_8[i]);
		new_session_kernel(session_16[i], graph_def_16[i]);
		new_session_kernel(session_32[i], graph_def_32[i]);
	}

	return 0;
}

int NNPredict::run_session(PRECISION* input2, PRECISION* output2, int input_dim, tensorflow::Session* run_sess)
{
	using namespace tensorflow;
	// Setup inputs and outputs:

	// Our graph doesn't require any inputs, since it specifies default values,
	// but we'll change an input to demonstrate.
	Tensor a(DT_FLOAT, TensorShape({ 1,input_dim }));
	auto input_tensor_mapped = a.tensor<PRECISION, 2>();
	//cout << 11 << endl;
	for (int x = 0; x < 1; x++)
	{
		for (int y = 0; y < input_dim; y++)
		{
			input_tensor_mapped(x, y) = input2[y];
		}
	}

	//cout << 11 << endl;
	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;
	std::string InputName = "Input/ref_pixel";
	std::string OutputName = "nn_pred";
	//std::string OutputName = "nn_pred";

	vector<std::pair<string, Tensor>> inputs;
	inputs.push_back(std::make_pair(InputName, a));

	//cout << 10 << endl;

	// Run the session, evaluating our "c" operation from the graph
	Status status = run_sess->Run(inputs, { OutputName }, {}, &outputs);

	//cout << 11 << endl;

	if (!status.ok()) {

		cout << input_dim << " status not ok" << endl;

		//std::cout << status.ToString() << "\n";
		return 1;
	}

	//cout << 1 << endl;

	// Grab the first output (we only evaluated one graph node: "c")
	// and convert the node to a scalar representation.
	tensorflow::Tensor* output = &outputs[0];
	const Eigen::TensorMap<Eigen::Tensor<PRECISION, 1, Eigen::RowMajor>, Eigen::Aligned>& prediction = output->flat<PRECISION>();
	const long count = prediction.size();
	for (int i = 0; i < count; ++i) {
		const PRECISION value = prediction(i);

		output2[i] = value;
		// value是该张量以一维数组表示时在索引i处的值。
	}

	// (There are similar methods for vectors and matrices here:
	// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

	// Print the results
	//std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
	//std::cout << output_c(0,0) << "\n"; // 30

	// Free any resources used by the session

	return 0;
}

void NNPredict::close_session()
{
	for (int i = 0; i < NUM_MODE; i++)
	{
		session_4[i]->Close();
		session_8[i]->Close();
		session_16[i]->Close();
		session_32[i]->Close();
	}
}

#pragma once


#define NOMINMAX
#define COMPILER_MSVC

#include "CommonDef.h"


#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"


#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/public/session.h"


#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

class NNPredict{

public:

tensorflow::GraphDef graph_def_4[NUM_MODE];
static tensorflow::Session* session_4[NUM_MODE];

tensorflow::GraphDef graph_def_8[NUM_MODE];
static tensorflow::Session* session_8[NUM_MODE];

tensorflow::GraphDef graph_def_16[NUM_MODE];
static tensorflow::Session* session_16[NUM_MODE];

tensorflow::GraphDef graph_def_32[NUM_MODE];
static tensorflow::Session* session_32[NUM_MODE];

static int a;

int load_pb_kernel(std::string model_add, tensorflow::GraphDef &graph_def);

int load_pb();

int new_session_kernel(tensorflow::Session* &session, tensorflow::GraphDef &graph_def);

int new_session();

int run_session(PRECISION* input2, PRECISION* output2, int input_dim, tensorflow::Session* run_sess);

void close_session();

int get_a() { return a; }

void set_a(int value) { a = value; }

};

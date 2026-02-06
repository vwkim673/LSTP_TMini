#define _CRT_SECURE_NO_WARNINGS
#define BOOST_ALL_NO_LIB
#define _USE_MATH_DEFINES
#define WIN32_LEAN_AND_MEAN

#pragma warning(disable: 4819 4996 4477 6258 4100)
#ifndef BOOST_USE_WINDOWS_H
#define BOOST_USE_WINDOWS_H
#endif

#define NOMINMAX
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <fstream>
#include <iterator>
#include <bitset>
#include <chrono>
#include <future>
#include <thread>
#include <filesystem>
#include <direct.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <condition_variable>
#include <atomic>
#include <signal.h>
#include <queue>
#include <ctime>
#include <iomanip>

#ifndef _WINDOW_SOCKET
#define _WINDOW_SOCKET
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <synchapi.h>
#endif

#include "VisionaryControl.h"
#include "CoLaParameterReader.h"
#include "CoLaParameterWriter.h"
#include "VisionaryTMiniData.h"    // Header specific for the Time of Flight data
#include "VisionaryDataStream.h"
#include "PointXYZ.h"
#include "PointCloudPlyWriter.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>

#include "miniINI.h"

#include <onnxruntime_cxx_api.h>

#include <boost/filesystem.hpp>

#ifndef PCL_COMMON_H
#define PCL_COMMON_H
#include <pcl/pcl_base.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>


#include <pcl/console/parse.h>
#include <pcl/common/random.h>
#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <pcl/common/intersections.h>
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/common/norms.h>
#include <pcl/common/time.h>
#include <pcl/common/distances.h>

#include <boost/thread.hpp>

// filter include
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>

// segmentation include
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// region growing
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>

//sample consensus for table calibrations
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_line.h>

#endif

#include <Eigen/Dense>

#include <open3d/Open3D.h>

#include "GlobalVar.h"
#include "DataStack.h"
//#include "VA.h"

#include "YOLO11.hpp"

#pragma comment(lib, "Dbghelp.lib")
#include <DbgHelp.h>

#define DEFAULT_RECVLEN 50
#define DEFAULT_SENDLEN 50

// Use the same default thresholds as Ultralytics CLI
const float confThreshold = 0.5f;  // Match Ultralytics default confidence threshold
const float iouThreshold = 0.45f;   // Match Ultralytics default IoU threshold

/*
* Update: 2025.08.19
* Update Log:
*	Added CLPS OK (chassis-container separation detection logic)
*/

const std::string program_version = "1.4";

std::vector<std::string> class_names = { "HOLE", "CONE", "LANDED", "GUIDE" };

void logMessage(const std::string& message);

struct bbx
{
	int label = 0;
	float x = 0;
	float y = 0;
	float center_x = 0;
	float center_y = 0;
	float w = 0;
	float h = 0;
	float prob = 0;
};

class DetectionAverager {
public:

	void update(const bbx& input) {
		count_++;

		avg_.label = input.label;

		avg_.x += (input.x - avg_.x) / count_;
		avg_.y += (input.y - avg_.y) / count_;
		avg_.w += (input.w - avg_.w) / count_;
		avg_.h += (input.h - avg_.h) / count_;
		avg_.center_x += (input.center_x - avg_.center_x) / count_;
		avg_.center_y += (input.center_y - avg_.center_y) / count_;
		avg_.prob += (input.prob - avg_.prob) / count_;
	}
	void set() {
		set_ = true;
	}

	bbx get_average() const {
		return avg_;
	}
	size_t get_count() const {
		return count_;
	}

	bool get_set() const {
		return set_;
	}

	void reset()
	{
		count_ = 0;
		avg_ = {};
		set_ = false;
	}

private:
	size_t count_;
	bbx avg_;

	bool set_;
};

class PCAAverager {
public:

	void update(const pcl::PointXYZ& input) {
		++count_;
		float k = 1.0f / static_cast<float>(count_);

		avg_.x += (input.x - avg_.x) * k;
		avg_.y += (input.y - avg_.y) * k;
		avg_.z += (input.z - avg_.z) * k;

	}
	void set() {
		set_ = true;
	}

	pcl::PointXYZ get_average() const {
		return avg_;
	}
	size_t get_count() const {
		return count_;
	}

	bool get_set() const
	{
		return set_;
	}

	void reset()
	{
		count_ = 0;
		avg_ = {};
		set_ = false;
	}

private:
	size_t count_ = 0;
	pcl::PointXYZ avg_{ 0.0f, 0.0f, 0.0f };
	bool set_;
};

struct IDEAL_POS
{
	bbx CONE;
	bbx LANDED;
	bbx GUIDE;

	pcl::PointXYZ CONE_PCA;
	pcl::PointXYZ LANDED_PCA;
	pcl::PointXYZ GUIDE_PCA;
};

struct JobInfo
{
	bool isMountCycle = false;
	bool isOffloadCycle = false;

	std::string jobPos = "";
	std::string chassisType = "";

	int chassisLen = 0;
	int jobSize = 0;

	IDEAL_POS preset;

	void set(bool mountCycle, bool offloadCycle,
		std::string job_pos, std::string chassis_type, int chassis_len, int job_size, IDEAL_POS preset_ip)
	{
		isMountCycle = mountCycle;
		isOffloadCycle = offloadCycle;
		jobPos = job_pos;
		chassisType = chassis_type;
		chassisLen = chassis_len;
		jobSize = job_size;
		preset = preset_ip;
	}

	void print_jobInfo()
	{
		logMessage("Job Info - MountCycle: " + std::to_string(isMountCycle) +
			", OffloadCycle: " + std::to_string(isOffloadCycle) +
			", JobPos: " + jobPos +
			", ChassisType: " + chassisType +
			", ChassisLen: " + std::to_string(chassisLen) +
			", JobSize: " + std::to_string(jobSize));

		/*
		if (chassisType == "XT")
		{
			logMessage("Job Info - Cone VA Preset: (" + std::to_string(preset.CONE.x) + "," + std::to_string(preset.CONE.y) + "," + std::to_string(preset.CONE.w) + "," + std::to_string(preset.CONE.h) + ")");
			logMessage("Job Info - Cone PCA Preset: (" + std::to_string(preset.CONE_PCA.x) + "," + std::to_string(preset.CONE_PCA.y) + "," + std::to_string(preset.CONE_PCA.z) + ")");
			logMessage("Job Info - Land VA Preset: (" + std::to_string(preset.LANDED.x) + "," + std::to_string(preset.LANDED.y) + "," + std::to_string(preset.LANDED.w) + "," + std::to_string(preset.LANDED.h) + ")");
			logMessage("Job Info - Land PCA Preset: (" + std::to_string(preset.LANDED_PCA.x) + "," + std::to_string(preset.LANDED_PCA.y) + "," + std::to_string(preset.LANDED_PCA.z) + ")");

		}
		else if (chassisType == "CST")
		{
			logMessage("Job Info - IGH VA Preset: (" + std::to_string(preset.GUIDE.x) + "," + std::to_string(preset.GUIDE.y) + "," + std::to_string(preset.GUIDE.w) + "," + std::to_string(preset.GUIDE.h) + ")");
			logMessage("Job Info - IGH PCA Preset: (" + std::to_string(preset.GUIDE_PCA.x) + "," + std::to_string(preset.GUIDE_PCA.y) + "," + std::to_string(preset.GUIDE_PCA.z) + ")");
		}
		*/
	}
};

struct ProcessResults
{
	//Chassis Type
	std::string VA_Chassis_Type = "";

	bool bDetected_XT = false;
	bool bDetected_CST = false;
	bool bDetected_Unknown = false;

	bool bDetected_Container = false;
	bool bDetected_Chassis = false;

	//Target Hole
	bbx VA_Target_Container;
	bbx VA_Target_Chassis;

	//Deviations
	int VA_Devout_X = 0;
	int VA_Devout_Y = 0;

	//VA Results
	bool VA_isLandout = false;
	bool VA_isLandOK = false;
	bool VA_isCLPS = false;
	bool VA_isCLPS_OK = false;

	//3D Results
	pcl::PointXYZ PCA_Target_Container;
	pcl::PointXYZ PCA_Target_Chassis;
	pcl::PointXYZ PCA_Target_Guide;

	//Deviations
	int PCA_Devout_X = 0;
	int PCA_Devout_Y = 0;
	int PCA_Devout_Z = 0;

	//3D Results
	bool PCA_isLandout = false;
	bool PCA_isLandOK = false;
	bool PCA_isCLPS = false;
	bool PCA_isCLPS_OK = false;

	bool bLandOutDetected = false;
	bool bLandOKDetected = false;
	bool bClpsDetected = false;
	bool bClpsOkDetected = false;
};

std::string app_path;

#pragma region INI Variables

std::string sensor_ip;
std::string sensor_port;

std::string SOCKET_IP = "127.0.0.1";
int SOCKET_PORT;

const int sendLen = 50;
const int recvLen = 50;

bool MODE_DEBUG = true;

std::string DEBUG_SAMPLE_JOB = "";

bool DEBUG_WITH_FILES = false;
std::string DEBUG_PATH;
std::string DEBUG_SENSOR_POSITION = "REAR_LEFT";

bool DEBUG_BATCH_JOB = false;
std::string DEBUG_BATCH_ROOT_DIR = "";
std::string DEBUG_BATCH_SAVE_DIR = "";

bool DEBUG_CONVERT_PCL_RANGE = false;

bool SEQUENTIAL_PROCESSING = false;
bool ENALBE_RESTART_APPLICATION = false;

std::string CURRENT_SENSOR_POSITION = "REAR_LEFT";

int PROB_LIMIT = 50;

std::string onnx_model_path;

int LANE_NUM_AS_LEFT = 0;
int LANE_NUM_AS_RIGHT = 0;

IDEAL_POS L_Pos;
IDEAL_POS R_Pos;
IDEAL_POS JOB_IP_Pos;

int G_DATA_LIMIT = 3000;
int T_DATA_LIMIT = 2000;

int LDO_NEAR_X_THRESHOLD = 50;
int LDO_NEAR_Y_THRESHOLD = 50;

int LDO_FAR_X_THRESHOLD = 50;
int LDO_FAR_Y_THRESHOLD = 50;

int LDO_GANTRY_THRESHOLD = 50;
int LDO_TROLLEY_THRESHOLD = 50;

int LDO_NCOUNT = 10;

int CLPS_NEAR_X_THRESHOLD = 50;
int CLPS_NEAR_Y_THRESHOLD = 50;

int CLPS_LOWER_X_THRESHOLD = 50;
int CLPS_LOWER_Y_THRESHOLD = 50;


int CLPS_NCOUNT = 10;

std::vector<std::string> IP_ADDRESSES = {};
#pragma endregion

#pragma region VisionaryTMini Variables
std::string current_lane_ip = "";

SOCKET connectSocket_ = NULL;
unsigned int heartbeat_, prev_heartbeat_;
int visionary_version_[4]{ 0 };
char block_n_[5]{};
char* block_;
int tzms_version_;

char* sendBuffer_;
char sendBuf_[DEFAULT_SENDLEN] = {};

char* recvBuffer_;
char recvBuf_[DEFAULT_RECVLEN] = {};

auto sck_connected = false;
//auto save_frame = false;

bool blnDataSave;
std::string SaveDir;
std::string SaveImageDir;
std::string SaveDepthDir;

//Job information
bool TZ_Offload_Cycle = false;
bool TZ_Mount_Cycle = false;

bool Job_Pos_B = false;
bool Job_Pos_S = false;
bool Job_Pos_C = false;

bool SPRD_TWL_Locked = false;
bool SPRD_TWL_Unlocked = false;
bool SPRD_Landed = false;

bool SPRD_20ft = false;
bool SPRD_40ft = false;
bool SPRD_45ft = false;

bool Target_20ft = false;
bool Target_40ft = false;
bool Target_45ft = false;

bool Hoist_Stopped = false;
bool Trolley_Stopped = false;
bool Gantry_Stopped = false;

bool Above_Safe_Height = false;

bool CHS_20ft, CHS_40ft, CHS_45ft;
bool CHS_XT, CHS_CST;

bool Logging_Enable = false;
bool Front_Left_Logging = false;
bool Front_Right_Logging = false;
bool Center_Left_Logging = false;
bool Center_Right_Logging = false;
bool Rear_Logging = false;

short Target_Height_Pos_mm = 0;
short Current_Hoist_Pos_mm = 0;
int Current_Trolley_Pos_mm = 0;
int Target_Trolley_Pos_mm = 0;

short intrim_Target_Height_Pos_mm = 0;
short intrim_Current_Hoist_Pos_mm = 0;
int intrim_Current_Trolley_Pos_mm = 0;
int intrim_Target_Trolley_Pos_mm = 0;

std::string saveName;
std::string saveDirName;

std::string job_name;
std::string job_result_folder_name;
//images, pointcloud, and .log files
std::string job_result_log_file_name;

bool sensor_connected = false;
bool sensor_fault = false;

std::chrono::system_clock::time_point sensor_last_attempted_time = std::chrono::system_clock::now();
std::chrono::system_clock::time_point sensor_last_connected_time = std::chrono::system_clock::now();

YOLO11Detector yolo_detector;

bool enable_stream = false;
bool enable_process = false;
bool enable_logging = false;

bool processThread_stopped = false;

bool ENABLE_SAVE_LOG_BIT = false;
bool enable_save_logs = false;

bool enabled_stream = false;
bool enabled_process = false;
bool enabled_logging = false;

bool trigger_log_delete = false;
bool trigger_terminate = false;

bool model_loaded = false;
bool model_initialized = false;

int tCntr_x, tCntr_y, tCntr_prob;
int tCone_x, tCone_y, tCone_prob;
int chassis_detected_type;

bool detected_xt = false;
bool detected_cst = false;
bool detected_chassis_type_unknown = false;

bool landout_detected = false;
bool landok_detected = false;
bool clps_detected = false;
bool clps_ok_detected = false;

int LDO_Current_Count = 0;
int LandOK_Current_Count = 0;
int CLPS_Current_Count = 0;
int CLPS_OK_Current_Count = 0;

int target_lane_number = 1;

int devOut_x, devOut_y;
int devOut_LDO_x, devOut_LDO_y;
//2025.03.11
int devOut_x_mm, devOut_y_mm;
int devOut_pca_x, devOut_pca_y, devOut_pca_z;

bool landout_detected_pca = false;
int landout_current_count_pca = 0;

bool landok_detected_pca = false;
int landok_current_count_pca = 0;

bool clps_detected_pca = false;
int clps_current_count_pca = 0;
bool clps_ok_detected_pca = false;
int clps_ok_current_count_pca = 0;

bool clps_detected_pca_only = false;
bool clps_ok_detected_pca_only = false;
int clps_detected_pca_only_count = 0;
int clps_ok_detected_pca_only_count = 0;

bool bDetected_hole = false, bDetected_cone = false;

bool bLandout_Detected_Out = false;
bool bLandOK_Detected_Out = false;

//combined output.
bool bCLPS_Detected_Out = false;
bool bCLPS_OK_Detected_Out = false;

JobInfo current_job_info;
ProcessResults current_process_results;

DetectionAverager AVG_CONE;
DetectionAverager AVG_HOLE;

DetectionAverager OFFLOAD_CONE;

//for debugging.
int UPPER_DIFF_X = 0;
int UPPER_DIFF_Y = 0;
int LOWER_DIFF_X = 0;
int LOWER_DIFF_Y = 0;

bool bAvg_VA_Usable = true;
int inferenceFailCounter = 0;
std::vector<int> chassis_sep_height = {};

PCAAverager AVG_CONE_PCA;

bbx Offload_Hole_Base;
bbx Offload_Hole_intrim;
int Offload_Hole_Base_Count = 0;
bool Offload_Hole_Base_Set = false;

pcl::PointXYZ PCA_intrim;
pcl::PointXYZ PCA_Base;
int PCA_Base_Count = 0;
bool PCA_Base_Set = false;

bbx LDO_intrim;
bbx LDO_Base;
int LDO_Base_Count = 0;
bool LDO_Base_Set = false;

bbx CLPS_intrim;
bbx CLPS_Base;
int CLPS_Base_Count = 0;
bool CLPS_Base_Set = false;

#pragma endregion

//Debug
std::string DEBUG_IMG_PATH = "";
std::string DEBUG_PLY_PATH = "";
std::vector<std::string> DEBUG_IMG_FILES;
std::vector<std::string> DEBUG_PLY_FILES;
int DEBUG_CURRENT_INDEX = 0;
int DEBUG_MAX_INDEX = 0;

LONG WINAPI MyUnhandledExceptionFilter(EXCEPTION_POINTERS* ExceptionInfo) {
	HANDLE hFile = CreateFile(L"crash.dmp", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (hFile != INVALID_HANDLE_VALUE) {
		MINIDUMP_EXCEPTION_INFORMATION mdei;
		mdei.ThreadId = GetCurrentThreadId();
		mdei.ExceptionPointers = ExceptionInfo;
		mdei.ClientPointers = FALSE;

		MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(),
			hFile, MiniDumpNormal, &mdei, nullptr, nullptr);

		CloseHandle(hFile);
	}

	return EXCEPTION_EXECUTE_HANDLER;
}

void reset_jobVariables()
{
	current_job_info = JobInfo();
	current_process_results = ProcessResults();

	tCntr_x = -10000; tCntr_y = -10000; tCntr_prob = -10000;
	tCone_x = -10000; tCone_y = -10000; tCone_prob = -10000;
	detected_xt = false; detected_cst = false; detected_chassis_type_unknown = false;

	devOut_x = -10000; devOut_y = -10000;
	devOut_LDO_x = -10000; devOut_LDO_y = -10000;
	devOut_pca_x = -10000; devOut_pca_y = -10000; devOut_pca_z = -10000;

	LDO_Current_Count = 0; CLPS_Current_Count = 0; LandOK_Current_Count = 0;
	CLPS_OK_Current_Count = 0;

	landout_detected_pca = false;
	landout_current_count_pca = 0;
	landok_detected_pca = false;
	landok_current_count_pca = 0;

	clps_detected_pca = false;
	clps_current_count_pca = 0;
	clps_ok_detected_pca = false;
	clps_ok_current_count_pca = 0;

	clps_detected_pca_only = false;
	clps_ok_detected_pca_only = false;
	clps_detected_pca_only_count = 0;
	clps_ok_detected_pca_only_count = 0;

	bDetected_hole = false;
	bDetected_cone = false;

	AVG_CONE.reset();
	AVG_HOLE.reset();
	AVG_CONE_PCA.reset();

	OFFLOAD_CONE.reset();

	UPPER_DIFF_X = 0;
	UPPER_DIFF_Y = 0;
	LOWER_DIFF_X = 0;
	LOWER_DIFF_Y = 0;

	bAvg_VA_Usable = true;
	inferenceFailCounter = 0;

	chassis_sep_height = {};

	PCA_intrim.x = 0; PCA_intrim.y = 0; PCA_intrim.z = 0;
	PCA_Base.x = 0; PCA_Base.y = 0; PCA_Base.z = 0;
	PCA_Base_Count = 0; PCA_Base_Set = false;

	Offload_Hole_Base = bbx();
	Offload_Hole_intrim = bbx();
	Offload_Hole_Base_Count = 0; Offload_Hole_Base_Set = false;

	LDO_intrim = bbx();
	LDO_Base = bbx();
	LDO_Base_Count = 0; LDO_Base_Set = false;

	CLPS_intrim = bbx();
	CLPS_Base = bbx(); 

	CLPS_Base_Count = 0; CLPS_Base_Set = false;

	landout_detected = false;
	landok_detected = false;
	clps_detected = false;
	clps_ok_detected = false;

	bCLPS_Detected_Out = false;
	bCLPS_OK_Detected_Out = false;

	bLandOK_Detected_Out = false;
	bLandout_Detected_Out = false;
}

//job result logging related
bool save_trigger_by_landed = false;
bool save_trigger_by_TWL_Locked = false;

std::chrono::system_clock::time_point last_frame_get_time = std::chrono::system_clock::now();
bool blnGetNewFrame = false;
bool blnFrameFault = false;

std::atomic<bool> socket_running(true);

/*
 * Mutex and Conditional Variables
 *
 */

std::mutex mutex_tmini_ctrl;
std::condition_variable cond_tmini_ctrl;
//this one is for running/terminating thread.
std::atomic<bool> tmini_ctrl_running(true);
bool tmini_ctrl_flag = false;

std::mutex mutex_tmini_data_stream;
std::condition_variable cond_tmini_data_stream;
//this one is for running/terminating thread.
std::atomic<bool> tmini_data_stream_running(true);
bool tmini_data_stream_flag = false;

std::mutex mutex_processing;
std::condition_variable cond_processing;
//this one is for running/terminating thread.
std::atomic<bool> proc_running(true);

std::mutex mutex_seq_processing;
std::condition_variable cond_seq_processing;
//this one is for running/terminating thread.
std::atomic<bool> seq_proc_running(true);
bool seq_proc_flag = false;


std::mutex mutex_logging;
std::condition_variable cond_logging;
//this one is for running/terminating thread.
std::atomic<bool> logging_running(true);

/*
 * Log Timer Variables
 */

std::queue<std::string> logQueue;
std::mutex queueMutex;
std::condition_variable cvLog;
std::atomic<bool> log_running(true);

std::mutex logMutex; // Mutex to protect log file writing
std::string currentLogFileName; // To track the current log file

std::queue<std::string> jobLogQueue;
std::mutex jlQueueMutex;
std::condition_variable cvJobLog;
std::atomic<bool> job_log_running(true);

std::mutex jobLogMutex; // Mutex to protect log file writing
std::string currentJobLogFileName; // To track the current log file

std::mutex jobLogEnableMutex;
std::condition_variable cond_jobLogEnable;

std::string appID = "";
std::string appName = "TMini";

// Generate Visionary instance
auto pDataHandler = std::make_shared<VisionaryTMiniData>();
VisionaryDataStream dataStream(pDataHandler);
VisionaryControl visionaryControl;

class ThreadSafeQueue
{
private:
	std::queue <std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::vector<uint16_t>, bool, std::string, std::chrono::system_clock::time_point>> queue;
	std::mutex mtx;
	std::condition_variable cv;

public:
	void push(std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::vector<uint16_t>, bool, std::string, std::chrono::system_clock::time_point> value) {
		std::lock_guard<std::mutex> lock(mtx);
		queue.push(value);
		cv.notify_one(); // Notify the consumer
	}

	// Pop element from the queue (blocks if empty)
	std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::vector<uint16_t>, bool, std::string, std::chrono::system_clock::time_point> pop() {
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [this] { return !queue.empty(); }); // Wait until queue is non-empty
		std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::vector<uint16_t>, bool, std::string, std::chrono::system_clock::time_point> value = queue.front();
		queue.pop();
		return value;
	}

	int GetQueueLen()
	{
		std::lock_guard<std::mutex> lock(mtx);
		return queue.size();
	}

	void clear() {
		std::lock_guard<std::mutex> lock(mtx);
		while (!queue.empty()) {
			queue.pop();
		}
	}
};

ThreadSafeQueue tsq;
DataStack dataStack;

bool flag = false;
bool saveFlag = false;

bool jobLogFlag = false;


void my_handler(int s)
{

	logging_running.store(false);
	proc_running.store(false);	
	socket_running.store(false);

	logMessage("Caught signal " + std::to_string(s));
	logMessage("Terminating tmini_process application triggered by Ctrl+C");

	log_running.store(false);

	exit(1);
}

std::vector<std::string> split(const std::string& str, char delimiter) {
	std::vector<std::string> tokens;
	std::stringstream ss(str);
	std::string token;

	while (std::getline(ss, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}

auto print_time() -> std::string
{
	auto now = std::chrono::system_clock::now();

	// Convert to time_t to use with std::localtime
	std::time_t current_time = std::chrono::system_clock::to_time_t(now);

	// Convert to tm struct (local time)
	std::tm local_time = *std::localtime(&current_time);

	// Get the milliseconds part
	auto duration = now.time_since_epoch();
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

	std::ostringstream oss;
	oss << std::put_time(&local_time, "%Y%m%d_%H%M%S") << "_" << std::setw(3) << std::setfill('0') << milliseconds;

	// Get the formatted time as a string
	std::string formatted_time = oss.str();

	// Output formatted current time
	return formatted_time;
}

//Check if directory exists
auto dirExists(const char* const path) -> int
{
	struct stat info;

	int statRC = stat(path, &info);
	if (statRC != 0)
	{
		if (errno == ENOENT) { return 0; } // something along the path does not exist
		if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
		return -1;
	}

	return (info.st_mode & S_IFDIR) ? 1 : 0;
}
//Check if directory exists, if not create one (returns true if created).
auto createDirectory_ifexists(std::string path) -> bool
{
	auto status = false;
	auto dirE = dirExists(path.c_str());
	if (dirE == 0)
	{
		auto ret = _mkdir(path.c_str());
		if (ret == -1) {
			logMessage("Error in creating directory!");
		}
		status = true;
	}
	return status;
}

// Get the current date as a string in the format YYYY-MM-DD
std::string getCurrentDate() {
	auto now = std::chrono::system_clock::now();
	std::time_t current_time = std::chrono::system_clock::to_time_t(now);
	std::tm local_time = *std::localtime(&current_time);
	std::ostringstream oss;
	oss << std::put_time(&local_time, "%Y%m%d");
	return oss.str();
}

// Function to write log entries to file
void logWriterThread() {
	std::ofstream logFile;

	createDirectory_ifexists(app_path + "/Log");

	while (log_running || !logQueue.empty()) {
		try
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			cvLog.wait(lock, [] { return !logQueue.empty() || !log_running; });

			// Get the current date and check if it has changed
			std::string newLogFileName = app_path + "/Log/" + appName + "_log_" + getCurrentDate() + ".log";

			// If the date has changed, close the current log file and open a new one
			if (newLogFileName != currentLogFileName) {
				if (logFile.is_open()) {
					logFile.close(); // Close the previous file
				}
				logFile.open(newLogFileName, std::ios::out | std::ios::app); // Open new log file
				currentLogFileName = newLogFileName; // Update the current log file name
			}

			// Write all logs in the queue to the current file
			while (!logQueue.empty()) {
				logFile << logQueue.front() << std::endl;
				logQueue.pop();
			}
			lock.unlock();
		}
		catch (std::exception& ex)
		{
			std::cout << "Error in log thread : " << std::string(ex.what());
			//return false;
		}
		catch (...)
		{
			std::cout << "Unknown error in log thread";
			//return false;
		}
	}

	// Close the log file at the end
	if (logFile.is_open()) {
		logFile.close();
	}
}

void jobLogWriterThread() {
	std::ofstream logFile;

	while (job_log_running || !jobLogQueue.empty()) {
		std::unique_lock<std::mutex> lock(jlQueueMutex);
		cvJobLog.wait(lock, [] { return !jobLogQueue.empty() || !job_log_running; });

		// Get the current date and check if it has changed
		std::string newLogFileName = job_result_log_file_name;
		
		// If the date has changed, close the current log file and open a new one
		if (newLogFileName != currentJobLogFileName) 
		{
			logMessage("New filename: " + newLogFileName);
			if (logFile.is_open()) {
				logFile.close(); // Close the previous file
			}
			logFile.open(newLogFileName, std::ios::out | std::ios::app); // Open new log file
			currentJobLogFileName = newLogFileName; // Update the current log file name
		}

		// Write all logs in the queue to the current file
		while (!jobLogQueue.empty()) {
			logFile << jobLogQueue.front() << std::endl;
			jobLogQueue.pop();
		}
		lock.unlock();
	}

	// Close the log file at the end
	if (logFile.is_open()) {
		logFile.close();
	}
}

// Function to log messages (called by other threads)
void logMessage(const std::string& message) {
	
	try
	{
		//std::cout << "Logging working " << message << "\n";
		std::stringstream ss;
		auto now = std::chrono::system_clock::now();
		auto time_since_epoch = now.time_since_epoch();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch).count() % 1000;

		std::time_t current_time = std::chrono::system_clock::to_time_t(now);
		std::tm local_time = *std::localtime(&current_time);

		ss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms << " - " << message;

		{
			std::lock_guard<std::mutex> lock(queueMutex);
			logQueue.push(ss.str());
		}
		printf((ss.str() + "\n").c_str());
		cvLog.notify_one();
	}
	catch (std::exception& ex)
	{
		std::cout << "Error in log message : " << std::string(ex.what());
		//return false;
	}
}
void jobLogMessage(const std::string& message) {
	std::stringstream ss;
	auto now = std::chrono::system_clock::now();
	auto time_since_epoch = now.time_since_epoch();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch).count() % 1000;

	std::time_t current_time = std::chrono::system_clock::to_time_t(now);
	std::tm local_time = *std::localtime(&current_time);

	ss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms << " - " << message;

	{
		std::lock_guard<std::mutex> lock(jlQueueMutex);
		jobLogQueue.push(ss.str());
	}
	//printf((ss.str() + "\n").c_str());
	cvJobLog.notify_one();
}

template <typename Duration>
auto print_time(tm t, Duration fraction) -> std::string
{
	using namespace std::chrono;
	char val[256];
	std::sprintf(val, "%04u%02u%02u_%02u%02u%02u_%03u", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, static_cast<unsigned>(fraction / milliseconds(1)));
	std::string s(val);
	return s;
}
//Represent current datetime as filename
auto time_as_name() -> std::string
{
	//name is specific to be time. yyyyMMdd_hhMMss_msec
	std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
	std::chrono::system_clock::duration tp = now.time_since_epoch();
	tp -= std::chrono::duration_cast<std::chrono::seconds>(tp);
	time_t tt = std::chrono::system_clock::to_time_t(now);

	std::string t = print_time(*localtime(&tt), tp);
	return t;
}
auto time_as_name(std::chrono::system_clock::time_point ts) -> std::string
{
	//name is specific to be time. yyyyMMdd_hhMMss_msec
	//std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
	auto now = ts;
	std::chrono::system_clock::duration tp = now.time_since_epoch();
	tp -= std::chrono::duration_cast<std::chrono::seconds>(tp);
	time_t tt = std::chrono::system_clock::to_time_t(now);

	std::string t = print_time(*localtime(&tt), tp);
	return t;
}

auto time_up_to_millseconds(std::chrono::system_clock::time_point time_data) -> std::string {
	std::chrono::system_clock::duration tp = time_data.time_since_epoch();
	tp -= std::chrono::duration_cast<std::chrono::seconds>(tp);
	time_t tt = std::chrono::system_clock::to_time_t(time_data);

	std::string t = print_time(*localtime(&tt), tp);
	return t;
}

auto parseAppName(string path) -> bool
{
	try
	{
		mINI::INIFile inireader(path + "/INI/seoho.ini");
		mINI::INIStructure inidata;

		inireader.read(inidata);

		appID = inidata.get("ID").get("APP");
		return true;
	}
	catch (std::exception& ex)
	{
		std::cout << "Error in parsing INI file : " << std::string(ex.what());
		return false;
	}
}
auto parseINI(string path) -> bool
{
	try
	{
		mINI::INIFile inireader(path + "/INI/seoho.ini");
		mINI::INIStructure inidata;

		inireader.read(inidata);

		appID = inidata.get("ID").get("APP");
		if (appID.find("LEFT") != std::string::npos)
		{
			CURRENT_SENSOR_POSITION = "REAR_LEFT";
			logMessage("Current application is for REAR LEFT sensor!");
		}
		else
		{
			CURRENT_SENSOR_POSITION = "REAR_RIGHT";
			logMessage("Current application is for REAR RIGHT sensor!");
		}
		//port
		//sensor_port = inidata.get("SENSOR").get("PORT");
		//logMessage("Port: " + sensor_port);

		//socket port
		SOCKET_PORT = std::stoi(inidata.get("SOCKET").get("SERVER_PORT"));
		logMessage("socket ip,port: " + SOCKET_IP + " , " + std::to_string(SOCKET_PORT));
		
		PROB_LIMIT = std::stoi(inidata.get("SYSTEM").get("PROB_LIMIT"));
		ENABLE_SAVE_LOG_BIT = inidata.get("SYSTEM").get("ENABLE_SAVE_LOG_BIT") == "1" ? true : false;

		SEQUENTIAL_PROCESSING = inidata.get("SYSTEM").get("SEQUENTIAL_PROCESSING") == "1" ? true : false;
		ENALBE_RESTART_APPLICATION = inidata.get("SYSTEM").get("ENABLE_RESTART_APPLICATION") == "1" ? true : false;

		onnx_model_path = inidata.get("MODEL").get("ONNX_PATH");
		logMessage("onnx model path: " + onnx_model_path);

		MODE_DEBUG = inidata.get("DEBUG").get("MODE_DEBUG") == "1" ? true : false;
		if (MODE_DEBUG) logMessage("Debug mode enabled!");

		DEBUG_SAMPLE_JOB = inidata.get("DEBUG").get("DEBUG_SAMPLE_JOB");

		DEBUG_CONVERT_PCL_RANGE = inidata.get("DEBUG").get("DEBUG_CONVERT_PCL_RANGE") == "1" ? true : false;

		DEBUG_SENSOR_POSITION = inidata.get("DEBUG").get("DEBUG_SENSOR_POSITION");
		
		DEBUG_WITH_FILES = inidata.get("DEBUG").get("DEBUG_WITH_FILES") == "1" ? true : false;

		DEBUG_PATH = inidata.get("DEBUG").get("DEBUG_PATH");

		DEBUG_BATCH_JOB = inidata.get("DEBUG").get("DEBUG_BATCH_JOB") == "1" ? true : false;
		if (DEBUG_BATCH_JOB) logMessage("Debug mode in batch mode!");
		DEBUG_BATCH_ROOT_DIR = inidata.get("DEBUG").get("DEBUG_BATCH_ROOT_DIR");
		DEBUG_BATCH_SAVE_DIR = inidata.get("DEBUG").get("DEBUG_BATCH_SAVE_DIR");

		if (MODE_DEBUG || DEBUG_WITH_FILES || DEBUG_BATCH_JOB) logMessage("Debug Sensor Pos: " + DEBUG_SENSOR_POSITION);

		logMessage("INI Configuration Complete!");
	}
	catch (std::exception& e)
	{
		logMessage("Error in parsing INI file : " + std::string(e.what()));
		return false;
	}
	return true;
}

void parseProcessINI(string path)
{
	mINI::INIFile file(path + "/INI/process.ini");
	mINI::INIStructure ini;

	if (file.read(ini))
	{
		try
		{
			G_DATA_LIMIT = std::stoi(ini.get("MEASURE_DISTANCE").get("GANTRY"));
			T_DATA_LIMIT = std::stoi(ini.get("MEASURE_DISTANCE").get("TROLLEY"));

			{ //Left Pos
				L_Pos.CONE.x = std::stoi(ini.get("IDEAL_POS_VA").get("L_CONE_X")) - 5;
				L_Pos.CONE.y = std::stoi(ini.get("IDEAL_POS_VA").get("L_CONE_Y")) - 5;
				L_Pos.CONE.w = std::stoi(ini.get("IDEAL_POS_VA").get("L_CONE_W")) + 10;
				L_Pos.CONE.h = std::stoi(ini.get("IDEAL_POS_VA").get("L_CONE_H")) + 10;
				L_Pos.CONE.center_x = L_Pos.CONE.x + int(L_Pos.CONE.w / 2);
				L_Pos.CONE.center_y = L_Pos.CONE.y + int(L_Pos.CONE.h / 2);

				L_Pos.LANDED.x = std::stoi(ini.get("IDEAL_POS_VA").get("L_LANDED_X")) - 5;
				L_Pos.LANDED.y = std::stoi(ini.get("IDEAL_POS_VA").get("L_LANDED_Y")) - 5;
				L_Pos.LANDED.w = std::stoi(ini.get("IDEAL_POS_VA").get("L_LANDED_W")) + 10;
				L_Pos.LANDED.h = std::stoi(ini.get("IDEAL_POS_VA").get("L_LANDED_H")) + 10;
				L_Pos.LANDED.center_x = L_Pos.LANDED.x + int(L_Pos.LANDED.w / 2);
				L_Pos.LANDED.center_y = L_Pos.LANDED.y + int(L_Pos.LANDED.h / 2);

				L_Pos.GUIDE.x = std::stoi(ini.get("IDEAL_POS_VA").get("L_GUIDE_X")) - 5;
				L_Pos.GUIDE.y = std::stoi(ini.get("IDEAL_POS_VA").get("L_GUIDE_Y")) - 5;
				L_Pos.GUIDE.w = std::stoi(ini.get("IDEAL_POS_VA").get("L_GUIDE_W")) + 10;
				L_Pos.GUIDE.h = std::stoi(ini.get("IDEAL_POS_VA").get("L_GUIDE_H")) + 10;
				L_Pos.GUIDE.center_x = L_Pos.GUIDE.x + int(L_Pos.GUIDE.w / 2);
				L_Pos.GUIDE.center_y = L_Pos.GUIDE.y + int(L_Pos.GUIDE.h / 2);

				L_Pos.CONE_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("L_CONE_X"));
				L_Pos.CONE_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("L_CONE_Y"));
				L_Pos.CONE_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("L_CONE_Z"));

				L_Pos.LANDED_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("L_LANDED_X"));
				L_Pos.LANDED_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("L_LANDED_Y"));
				L_Pos.LANDED_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("L_LANDED_Z"));

				L_Pos.GUIDE_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("L_GUIDE_X"));
				L_Pos.GUIDE_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("L_GUIDE_Y"));
				L_Pos.GUIDE_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("L_GUIDE_Z"));
			}
			{ //Right Pos
				R_Pos.CONE.x = std::stoi(ini.get("IDEAL_POS_VA").get("R_CONE_X")) - 5;
				R_Pos.CONE.y = std::stoi(ini.get("IDEAL_POS_VA").get("R_CONE_Y")) - 5;
				R_Pos.CONE.w = std::stoi(ini.get("IDEAL_POS_VA").get("R_CONE_W")) + 10;
				R_Pos.CONE.h = std::stoi(ini.get("IDEAL_POS_VA").get("R_CONE_H")) + 10;
				R_Pos.CONE.center_x = R_Pos.CONE.x + int(R_Pos.CONE.w / 2);
				R_Pos.CONE.center_y = R_Pos.CONE.y + int(R_Pos.CONE.h / 2);

				R_Pos.LANDED.x = std::stoi(ini.get("IDEAL_POS_VA").get("R_LANDED_X")) - 5;
				R_Pos.LANDED.y = std::stoi(ini.get("IDEAL_POS_VA").get("R_LANDED_Y")) - 5;
				R_Pos.LANDED.w = std::stoi(ini.get("IDEAL_POS_VA").get("R_LANDED_W")) + 10;
				R_Pos.LANDED.h = std::stoi(ini.get("IDEAL_POS_VA").get("R_LANDED_H")) + 10;
				R_Pos.LANDED.center_x = R_Pos.LANDED.x + int(R_Pos.LANDED.w / 2);
				R_Pos.LANDED.center_y = R_Pos.LANDED.y + int(R_Pos.LANDED.h / 2);

				R_Pos.GUIDE.x = std::stoi(ini.get("IDEAL_POS_VA").get("R_GUIDE_X")) - 5;
				R_Pos.GUIDE.y = std::stoi(ini.get("IDEAL_POS_VA").get("R_GUIDE_Y")) - 5;
				R_Pos.GUIDE.w = std::stoi(ini.get("IDEAL_POS_VA").get("R_GUIDE_W")) + 10;
				R_Pos.GUIDE.h = std::stoi(ini.get("IDEAL_POS_VA").get("R_GUIDE_H")) + 10;
				R_Pos.GUIDE.center_x = R_Pos.GUIDE.x + int(R_Pos.GUIDE.w / 2);
				R_Pos.GUIDE.center_y = R_Pos.GUIDE.y + int(R_Pos.GUIDE.h / 2);

				R_Pos.CONE_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("R_CONE_X"));
				R_Pos.CONE_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("R_CONE_Y"));
				R_Pos.CONE_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("R_CONE_Z"));

				R_Pos.LANDED_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("R_LANDED_X"));
				R_Pos.LANDED_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("R_LANDED_Y"));
				R_Pos.LANDED_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("R_LANDED_Z"));

				R_Pos.GUIDE_PCA.x = std::stof(ini.get("IDEAL_POS_PCA").get("R_GUIDE_X"));
				R_Pos.GUIDE_PCA.y = std::stof(ini.get("IDEAL_POS_PCA").get("R_GUIDE_Y"));
				R_Pos.GUIDE_PCA.z = std::stof(ini.get("IDEAL_POS_PCA").get("R_GUIDE_Z"));
			}

			//Landout-VA
			LDO_NEAR_X_THRESHOLD = std::stoi(ini.get("LANDOUT").get("NEAR_X_THRESHOLD"));
			LDO_NEAR_Y_THRESHOLD = std::stoi(ini.get("LANDOUT").get("NEAR_Y_THRESHOLD"));

			LDO_FAR_X_THRESHOLD = std::stoi(ini.get("LANDOUT").get("FAR_X_THRESHOLD"));
			LDO_FAR_Y_THRESHOLD = std::stoi(ini.get("LANDOUT").get("FAR_Y_THRESHOLD"));

			LDO_GANTRY_THRESHOLD = std::stoi(ini.get("LANDOUT").get("GANTRY_THRESHOLD_PCA"));
			LDO_TROLLEY_THRESHOLD = std::stoi(ini.get("LANDOUT").get("TROLLEY_THRESHOLD_PCA"));

			LDO_NCOUNT = std::stoi(ini.get("LANDOUT").get("NCOUNT"));

			//CLPS-VA
			CLPS_NEAR_X_THRESHOLD = std::stoi(ini.get("CLPS").get("NEAR_X_THRESHOLD"));
			CLPS_NEAR_Y_THRESHOLD = std::stoi(ini.get("CLPS").get("NEAR_Y_THRESHOLD"));

			CLPS_LOWER_X_THRESHOLD = std::stoi(ini.get("CLPS").get("LOWER_X_THRESHOLD"));
			CLPS_LOWER_Y_THRESHOLD = std::stoi(ini.get("CLPS").get("LOWER_Y_THRESHOLD"));

			CLPS_NCOUNT = std::stoi(ini.get("CLPS").get("NCOUNT"));

			logMessage("PROCESS INI Configuration Complete!");
		}
		catch (std::exception& ex)
		{
			logMessage("Error: " + std::string(ex.what()) + " from reading INI/process.ini");
		}
		catch (...)
		{
			logMessage("Unknown error reading from INI/process.ini");
		}
	}
	else
	{
		logMessage("Failed to read INI/process.ini file");
	}
}

void parseIPINI(string path)
{
	mINI::INIFile file(path + "/INI/IP_LIST.ini");
	mINI::INIStructure ini;
	if (file.read(ini))
	{
		try
		{
			if (appID == "TMINI_LEFT")
			{
				for (int i = 0; i < 6; i++)
				{
					auto ip = ini.get("LEFT_IP").get("LANE" + std::to_string(i + 1));
					if (ip != "") {
						IP_ADDRESSES.push_back(ip);
					}
				}
			}
			else if (appID == "TMINI_RIGHT")
			{
				for (int i = 0; i < 6; i++)
				{
					auto ip = ini.get("RIGHT_IP").get("LANE" + std::to_string(i + 1));
					if (ip != "") {
						IP_ADDRESSES.push_back(ip);
					}
				}
				
			}
			logMessage("IP INI Configuration Complete!");
		}
		catch (std::exception& ex)
		{
			logMessage("Error: " + std::string(ex.what()) + " from reading INI/IP_LIST.ini");
		}
		catch (...)
		{
			logMessage("Unknown error reading from INI/IP_LIST.ini");
		}
	}
	else
	{
		logMessage("Failed to read INI/IP_LIST.ini file");
	}
}

std::wstring s2ws(const std::string& s) //string -> LPCWSTR needs first conversion to wstring then to LPCWSTR.
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}

template <size_t N1, size_t N2>
std::bitset <N1 + N2> concat(const std::bitset <N1>& b1, const std::bitset <N2>& b2) {
	std::string s1 = b1.to_string();
	std::string s2 = b2.to_string();
	return std::bitset <N1 + N2>(s1 + s2);
}

template<std::size_t B>
long bitset_to_long(const std::bitset<B>& b) {
	struct { long x : B; } s;
	return s.x = b.to_ulong();
}

void reset_recvVariables()
{
	TZ_Offload_Cycle = false;
	TZ_Mount_Cycle = false;

	Job_Pos_B = false;
	Job_Pos_S = false;
	Job_Pos_C = false;

	Target_20ft = false;
	Target_40ft = false;
	Target_45ft = false;

	SPRD_TWL_Locked = false;
	SPRD_TWL_Unlocked = false;
	SPRD_Landed = false;

	SPRD_20ft = false;
	SPRD_40ft = false;
	SPRD_45ft = false;

	Hoist_Stopped = false;
	Trolley_Stopped = false;
	Gantry_Stopped = false;

	Above_Safe_Height = false;

	Logging_Enable = false;
	Front_Left_Logging = false;
	Front_Right_Logging = false;
	Center_Left_Logging = false;
	Center_Right_Logging = false;
	Rear_Logging = false;

	Target_Height_Pos_mm = 0;
	Current_Hoist_Pos_mm = 0;
	Current_Trolley_Pos_mm = 0;
	Target_Trolley_Pos_mm = 0;

	intrim_Target_Height_Pos_mm = 0;
	intrim_Current_Hoist_Pos_mm = 0;
	intrim_Current_Trolley_Pos_mm = 0;
	intrim_Target_Trolley_Pos_mm = 0;

	enable_stream = false;
	enable_process = false;
	enable_logging = false;

}

std::vector<std::string> getAllFiles(boost::filesystem::path const& root, std::string const& ext)
{
	std::vector<boost::filesystem::path> paths;
	std::vector<std::string> full_paths;

	if (boost::filesystem::exists(root) && boost::filesystem::is_directory(root))
	{
		for (auto const& entry : boost::filesystem::recursive_directory_iterator(root))
		{
			if (boost::filesystem::is_regular_file(entry) && entry.path().extension() == ext)
			{
				paths.emplace_back(entry.path().filename());
				full_paths.push_back(entry.path().string());
			}
		}
	}
	return full_paths;
}

std::vector<std::string> ListSubDirectories(const std::string& directoryPath) {
	std::vector<std::string> subDirs;

	boost::filesystem::path dir(directoryPath);
	if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)) {
		// Iterate through the directory
		for (boost::filesystem::directory_entry& entry : boost::filesystem::directory_iterator(dir)) {
			if (boost::filesystem::is_directory(entry.status())) {
				subDirs.push_back(entry.path().string());
			}
		}
	}
	else {
		std::cerr << "Provided path is not a valid directory: " << directoryPath << std::endl;
	}

	return subDirs;
}

auto makeCommand(char* sendbuf) -> void {
	try
	{
		//send heartbeat
			//increment heartbeat
		heartbeat_ += 1;
		if (heartbeat_ > 65535) heartbeat_ = 0;
		//std::cout << "Sending heartbeat: " << std::to_string(heartbeat_) << std::endl;
		std::bitset<16>heartbeat_bit(heartbeat_);
		const std::string temp_str = heartbeat_bit.to_string();
		std::bitset<8>first_half(temp_str.substr(0, temp_str.size() / 2));
		std::bitset<8>second_half(temp_str.substr(temp_str.size() / 2, temp_str.size()));

		sendbuf[0] = static_cast<char>(first_half.to_ulong());
		sendbuf[1] = static_cast<char>(second_half.to_ulong());

		std::bitset<8>op_byte_2(0);
		if (model_loaded) op_byte_2.set(0);
		if (model_initialized) op_byte_2.set(1);
		if (sensor_connected) op_byte_2.set(2);
		if (sensor_fault) op_byte_2.set(3);
		if (enabled_stream) op_byte_2.set(4);
		if (enabled_process) op_byte_2.set(5);
		if (enabled_logging) op_byte_2.set(6);
		//

		sendbuf[2] = static_cast<char>(op_byte_2.to_ulong());

		
		//detected_cone_x
		{
			std::bitset<32>int_bitArr(current_process_results.VA_Target_Chassis.x);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[3] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[4] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[5] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[6] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//detected_cone_y
		{
			std::bitset<32>int_bitArr(current_process_results.VA_Target_Chassis.y);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[7] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[8] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[9] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[10] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//detected_cone_prob
		{
			std::bitset<32>int_bitArr(current_process_results.VA_Target_Chassis.prob);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[11] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[12] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[13] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[14] = static_cast<char>(fourth_quarter.to_ulong());
		}

		//detected_cntr_x
		{
			std::bitset<32>int_bitArr(current_process_results.VA_Target_Container.x);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[15] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[16] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[17] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[18] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//detected_cntr_y
		{
			std::bitset<32>int_bitArr(current_process_results.VA_Target_Container.y);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[19] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[20] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[21] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[22] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//detected_cntr_prob
		{
			std::bitset<32>int_bitArr(current_process_results.VA_Target_Container.prob);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[23] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[24] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[25] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[26] = static_cast<char>(fourth_quarter.to_ulong());
		}

		std::bitset<8>op_byte_27(0);
		if (current_process_results.bDetected_XT) op_byte_27.set(0);
		if (current_process_results.bDetected_CST) op_byte_27.set(1);
		if (current_process_results.bDetected_Unknown) op_byte_27.set(2);
		if (current_process_results.bLandOutDetected) op_byte_27.set(3);
		if (clps_detected) op_byte_27.set(4);
		if (clps_ok_detected) op_byte_27.set(5);
		if (current_process_results.bLandOKDetected) op_byte_27.set(6);
		//if (landout_detected_pca) op_byte_27.set(6);

		sendbuf[27] = static_cast<char>(op_byte_27.to_ulong());

		//dev out x
		{
			std::bitset<32>int_bitArr(current_process_results.VA_Devout_X);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[28] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[29] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[30] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[31] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//dev out y
		{
			std::bitset<32>int_bitArr(current_process_results.VA_Devout_Y);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[32] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[33] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[34] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[35] = static_cast<char>(fourth_quarter.to_ulong());
		}

		//dev out pca x
		{
			std::bitset<32>int_bitArr(current_process_results.PCA_Devout_X);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[36] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[37] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[38] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[39] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//dev out pca y
		{
			std::bitset<32>int_bitArr(current_process_results.PCA_Devout_Y);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[40] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[41] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[42] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[43] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//dev out pca z
		{
			std::bitset<32>int_bitArr(current_process_results.PCA_Devout_Z);
			const std::string temp_str = int_bitArr.to_string();
			std::bitset<8>first_quarter(temp_str.substr(0, 8));
			std::bitset<8>second_quarter(temp_str.substr(8, 16));
			std::bitset<8>third_quarter(temp_str.substr(16, 24));
			std::bitset<8>fourth_quarter(temp_str.substr(24, 32));

			sendbuf[44] = static_cast<char>(first_quarter.to_ulong());
			sendbuf[45] = static_cast<char>(second_quarter.to_ulong());
			sendbuf[46] = static_cast<char>(third_quarter.to_ulong());
			sendbuf[47] = static_cast<char>(fourth_quarter.to_ulong());
		}
		//dev out results
		{
			std::bitset<8>op_byte_48(0);
			if (landout_detected_pca) op_byte_48.set(0);
			if (clps_detected_pca) op_byte_48.set(1);
			if (clps_ok_detected_pca) op_byte_48.set(2);
			if (current_process_results.bDetected_Container) op_byte_48.set(3);
			if (current_process_results.bDetected_Chassis) op_byte_48.set(4);

			//2025.08.18
			if (clps_ok_detected_pca_only) op_byte_48.set(5);
			if (current_process_results.bClpsDetected) op_byte_48.set(6);
			if (current_process_results.bClpsOkDetected) op_byte_48.set(7);

			sendbuf[48] = static_cast<char>(op_byte_48.to_ulong());
		}
		
	}
	catch (std::exception& ex)
	{
		logMessage("Error in making command : " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("Unknown error in making command");
	}
}

std::string makeJobFolderName()
{
	std::string job_info = print_time() + std::string("_");
	if (TZ_Offload_Cycle) job_info.append("Offload_");
	else if (TZ_Mount_Cycle) job_info.append("Mount_");
	else job_info.append("None_");

	if (Job_Pos_B) job_info.append("Big_");
	else if (Job_Pos_S) job_info.append("Small_");
	else if (Job_Pos_C) job_info.append("Center_");
	else job_info.append("None_");

	if (CHS_XT) job_info.append("XT_");
	else if (CHS_CST) job_info.append("CST_");
	else job_info.append("None_");

	if (CHS_20ft) job_info.append("CHS20_");
	else if (CHS_40ft) job_info.append("CHS40_");
	else if (CHS_45ft) job_info.append("CHS45_");
	else job_info.append("None_");

	if (SPRD_20ft) job_info.append("SPRD20_");
	else if (SPRD_40ft) job_info.append("SPRD40_");
	else if (SPRD_45ft) job_info.append("SPRD45_");
	else job_info.append("None_");

	//2025.08.22 added target lane number

	job_info.append(std::to_string(target_lane_number) + "_");

	job_info.append(appName);

	//if (target_lane_number == LANE_NUM_AS_LEFT) job_info.append(SENSOR_POSITION + "_LEFT_");	
	//else if (target_lane_number == LANE_NUM_AS_RIGHT) job_info.append(SENSOR_POSITION + "_RIGHT_");
	//else job_info.append(SENSOR_POSITION + "_");
	
	return job_info;
}

void parseCommand(char recvbuf[])
{
	try
	{
		char* recvBuf = &recvbuf[0];

		//heartbeat section
		{
			std::bitset<8> byte0(recvBuf[0]);
			std::bitset<8> byte1(recvBuf[1]);
			std::bitset<16> temp_heartbeat_ = concat(byte0, byte1);
			heartbeat_ = temp_heartbeat_.to_ulong();
			//std::cout << "Received heartbeat: " << std::to_string(heartbeat_) << std::endl;
		}
		{
			std::bitset<8> byte2(recvBuf[2]);
			TZ_Offload_Cycle = (bool)byte2[0];
			TZ_Mount_Cycle = (bool)byte2[1];

			Job_Pos_B = (bool)byte2[2];
			Job_Pos_S = (bool)byte2[3];
			Job_Pos_C = (bool)byte2[4];

			//Target_20ft = (bool)byte13[4];
			//Target_40ft = (bool)byte13[5];
			//Target_45ft = (bool)byte13[6];

			//std::bitset<8> byte14(recvBuf[14]);
			bool temp_twl_l = (bool)byte2[5];
			if (temp_twl_l != SPRD_TWL_Locked)
			{
				if (enabled_process)
				{
					save_trigger_by_TWL_Locked = true;
					jobLogMessage("Save Triggered by SPRD TWL Locked");
				}
			}
			SPRD_TWL_Locked = temp_twl_l;

			SPRD_TWL_Unlocked = (bool)byte2[6];

			bool temp = (bool)byte2[7];
			if (temp != SPRD_Landed)
			{
				if (enabled_process)
				{
					save_trigger_by_landed = true;
					jobLogMessage("Save Triggered by SPRD Landed");
				}
			}
			SPRD_Landed = temp;

			std::bitset<8> byte3(recvBuf[3]);
			SPRD_20ft = (bool)byte3[0];
			SPRD_40ft = (bool)byte3[1];
			SPRD_45ft = (bool)byte3[2];

			Hoist_Stopped = (bool)byte3[3];
			Trolley_Stopped = (bool)byte3[4];
			Gantry_Stopped = (bool)byte3[5];

			Above_Safe_Height = (bool)byte3[6];

			std::bitset<8> byte4(recvBuf[4]);
			CHS_20ft = (bool)byte4[0];
			CHS_40ft = (bool)byte4[1];
			CHS_45ft = (bool)byte4[2];

			CHS_XT = (bool)byte4[3];
			CHS_CST = (bool)byte4[4];
		}
		{
			
			std::bitset<8> byte6(recvBuf[6]);
			std::bitset<8> byte7(recvBuf[7]);
			std::bitset<16> temp = concat(byte6, byte7);
			Target_Height_Pos_mm = bitset_to_long(temp);

			std::bitset<8> byte8(recvBuf[8]);
			std::bitset<8> byte9(recvBuf[9]);
			std::bitset<16> temp2 = concat(byte8, byte9);
			Current_Hoist_Pos_mm = bitset_to_long(temp2);

			std::bitset<8> byte10(recvBuf[10]);
			std::bitset<8> byte11(recvBuf[11]);
			std::bitset<8> byte12(recvBuf[12]);
			std::bitset<8> byte13(recvBuf[13]);

			std::bitset<16> byte1011 = concat(byte10, byte11);
			std::bitset<16> byte1213 = concat(byte12, byte13);

			std::bitset<32> byte1234 = concat(byte1011, byte1213);
			Current_Trolley_Pos_mm = static_cast<signed long>(byte1234.to_ulong());

			std::bitset<8> byte14(recvBuf[14]);
			std::bitset<8> byte15(recvBuf[15]);
			std::bitset<8> byte16(recvBuf[16]);
			std::bitset<8> byte17(recvBuf[17]);

			std::bitset<16> byte1415 = concat(byte14, byte15);
			std::bitset<16> byte1617 = concat(byte16, byte17);

			std::bitset<32> byte4567 = concat(byte1415, byte1617);
			Target_Trolley_Pos_mm = static_cast<signed long>(byte4567.to_ulong());
			
		}

		{
			std::bitset<8> byte20(recvBuf[20]);
			std::bitset<8> byte21(recvBuf[21]);
			std::bitset<16> temp = concat(byte20, byte21);
			auto tempVal = bitset_to_long(temp);
			if (tempVal != target_lane_number)
			{
				logMessage("Target Lane Number Changed from " + std::to_string(target_lane_number) + " to " + std::to_string(tempVal));
			}
			target_lane_number = tempVal;
		}

		{
			std::bitset<8> byte18(recvBuf[18]);
			enable_stream = (bool)byte18[0];
			
			bool tempVal = (bool)byte18[1];
			if (tempVal && tempVal != enable_process)
			{
				//rising edge.
				logMessage("Enable Process Detected!");

				if (target_lane_number > 0 && target_lane_number < 7)
				{
					current_lane_ip = IP_ADDRESSES[target_lane_number - 1];
					logMessage("Current Lane IP: " + current_lane_ip);
				}
				else
				{
					logMessage("Invalid Lane Number Detected! Setting it to default=1 and reporting app fault");
					current_lane_ip = IP_ADDRESSES[0];
					logMessage("Current Lane IP: " + current_lane_ip);
				}

				if (!SEQUENTIAL_PROCESSING)
				{
					dataStack.Clear_Stack();
					//trigger tmini setup and stream.
					{
						std::unique_lock<std::mutex> lock(mutex_tmini_ctrl);
						tmini_ctrl_flag = true;
					}
					cond_tmini_ctrl.notify_one();				
				}
				
				//streaming thread will be triggered from tmini_ctrl thread.
				reset_jobVariables();

				//Create necessary job result folders and specify names.
				job_result_folder_name = std::string("JOB_LOG/");
				createDirectory_ifexists(job_result_folder_name);
				job_result_folder_name += std::string("Lane") + std::to_string(target_lane_number) + "/";
				createDirectory_ifexists(job_result_folder_name);

				job_name = makeJobFolderName();
				logMessage("Job: " + job_name);
				job_result_folder_name.append(job_name);
				job_result_folder_name.append("_JOBLOG");
				createDirectory_ifexists(job_result_folder_name);

				//images, pointcloud
				createDirectory_ifexists(job_result_folder_name + "/Image");
				createDirectory_ifexists(job_result_folder_name + "/Depth");
				//job info log.
				job_result_log_file_name = job_result_folder_name + "/" + job_name + "_job_info.log";
				//allow job result log thread to resume?

				logMessage(job_result_log_file_name);
				jobLogMessage("Testing if Job Log is created on each job start.");
				
				if (CURRENT_SENSOR_POSITION.find("LEFT") != std::string::npos) JOB_IP_Pos = L_Pos;
				else JOB_IP_Pos = R_Pos;
				
				//Update Job Info
				std::string job_pos = "Center";
				if (Job_Pos_B) job_pos = "Big";
				else if (Job_Pos_S) job_pos = "Small";

				std::string chassis_type = "XT";
				if (CHS_CST) chassis_type = "CST";

				int chassis_len = 40;
				if (CHS_20ft) chassis_len = 20;
				else if (CHS_45ft) chassis_len = 45;

				int job_size = 40;
				if (SPRD_20ft) job_size = 20;
				else if (SPRD_45ft) job_size = 45;

				current_job_info.set(TZ_Mount_Cycle, TZ_Offload_Cycle,
					job_pos, chassis_type, chassis_len, job_size, JOB_IP_Pos);
				current_job_info.print_jobInfo();

				enabled_process = true;

				if (SEQUENTIAL_PROCESSING)
				{
					seq_proc_flag = true;
					//wake seq processing thread.
					cond_seq_processing.notify_one();
				}
				else
				{
					proc_running.store(true);
					//Wake the processing thread.
					flag = true;
					cond_processing.notify_one();
				}
				trigger_log_delete = false;
				trigger_terminate = false;

				logMessage("Process Enabled for Lane: " + std::to_string(target_lane_number) + " with job name: " + job_name);
			}
			else if (!tempVal && tempVal != enable_process)
			{
				logMessage("Disable Process Detected!");
				
				if (SEQUENTIAL_PROCESSING)
				{
					seq_proc_flag = false;
					//wake seq processing thread.
					cond_seq_processing.notify_one();
				}
				else
				{
					flag = false;

					tmini_ctrl_flag = false;
					//streaming off.
					tmini_data_stream_flag = false;

					enabled_process = false;
				}
				
				if (!enable_save_logs && ENABLE_SAVE_LOG_BIT)
				{
					logMessage("Delete Log Triggered.");
					trigger_log_delete = true;
					//need to delete "saved" logs.
				}

				reset_jobVariables();

				trigger_terminate = true;

				//wake data save thread
				//saveFlag = true;
				//cond_logging.notify_one();
			}
			enable_process = tempVal;

			//sensor re-connection check.
			if (enable_process &&!SEQUENTIAL_PROCESSING)
			{
				if (sensor_fault && !sensor_connected) //comm fault.
				{
					auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - sensor_last_attempted_time).count();
					if (ts > 10000) //10 seconds
					{
						logMessage("Sensor Connection Fault Detected! Attempting to reconnect...");
						sensor_connected = false;
						sensor_fault = false;
						{
							std::unique_lock<std::mutex> lock(mutex_tmini_ctrl);
							tmini_ctrl_flag = true; //wake tmini ctrl thread.
						}
						cond_tmini_ctrl.notify_one();
					}
				}
			}

			if (ENABLE_SAVE_LOG_BIT && trigger_log_delete && !enable_logging)
			{
				logMessage("Deleting save logs as per command.");

				//stop any files on queue
				if (tsq.GetQueueLen() > 0)
				{
					tsq.clear();
					logMessage("Cleared Thread Safe Queue.");
				}

				//delete folder recursively.
				//saveDirName
				try
				{
					if (std::filesystem::exists(saveDirName)) {
						std::filesystem::remove_all(saveDirName);  // Recursively removes the directory and its contents
						logMessage("Directory removed: " + saveDirName);
					}
					else
					{
						logMessage("Directory does not exist: " + saveDirName);
					}
				}
				catch (const std::filesystem::filesystem_error& e) {
					logMessage("Filesystem error: " + std::string(e.what()));
				}

				trigger_log_delete = false;
			}

			tempVal = (bool)byte18[2];
			if (tempVal && tempVal != enable_logging)
			{
				//rising edge.
				logMessage("Enable Logging Detected!");

				saveDirName = std::string("SAVE/");
				createDirectory_ifexists(saveDirName);
				saveDirName += std::string("Lane") + std::to_string(target_lane_number) + "/";
				createDirectory_ifexists(saveDirName);

				auto job_info = makeJobFolderName();
				saveDirName.append(job_info);

				logMessage(saveDirName);

				//createDirectory_ifexists("SAVE");
				createDirectory_ifexists(saveDirName);
				createDirectory_ifexists(saveDirName + "/Image");
				createDirectory_ifexists(saveDirName + "/Depth");
				//createDirectory_ifexists(saveDirName + "/DistMap");

				enabled_logging = true;

				//wake data save thread
				saveFlag = true;
				cond_logging.notify_one();

				enable_logging = true;

			}
			else if (!tempVal && tempVal != enable_logging)
			{
				logMessage("Disable Logging Detected!");
				//tmini_ctrl_flag = false;
				//streaming off.
				//tmini_data_stream_flag = false;

				saveFlag = false;
				enable_logging = false;
				enabled_logging = false;
			}

			enable_save_logs = (bool)byte18[4];

			if (ENALBE_RESTART_APPLICATION)
			{
				if (trigger_terminate && tsq.GetQueueLen() == 0)
				{
					logMessage("Triggered Application Termination!");

					//Nothing to save.	
					proc_running.store(false);
					cond_processing.notify_one();

					logging_running.store(false);
					cond_logging.notify_one();

					tmini_ctrl_running.store(false);
					cond_tmini_ctrl.notify_one();

					tmini_data_stream_running.store(false);
					cond_tmini_data_stream.notify_one();

					socket_running.store(false);

					logMessage("Threads to be terminated.");
				}
			}
		}
		
		return;
	}
	catch (std::exception& ex)
	{
		logMessage("Error: " + std::string(ex.what()) + " from parsing command.");
		return;
	}
	catch (...)
	{
		logMessage("Unknown error from parsing command.");
		return;
	}
}

auto controlConnected = false;
auto dataConnected = false;

namespace cc_detail {

	inline void normalizePlane(Eigen::Vector4f& p) {
		Eigen::Vector3f n = p.head<3>();
		float norm = n.norm();
		if (norm > 1e-9f) p /= norm;
	}

	// Constrained plane fit: plane normal parallel to `axis` (within eps_angle).
	inline bool fitParallelPlane(
		const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
		const Eigen::Vector3f& axis,
		float eps_angle_rad,
		float dist_thresh,
		int max_iters,
		pcl::ModelCoefficients& coeff_out,
		pcl::PointIndices& inliers_out)
	{
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setAxis(axis);
		seg.setEpsAngle(eps_angle_rad);
		seg.setDistanceThreshold(dist_thresh);
		seg.setMaxIterations(max_iters);
		seg.setInputCloud(cloud);
		seg.segment(inliers_out, coeff_out);
		return !inliers_out.indices.empty() && coeff_out.values.size() >= 4;
	}

	// Intersection line of two planes: n1·x + d1 = 0 and n2·x + d2 = 0
	inline bool intersectTwoPlanes(
		const Eigen::Vector4f& p1,
		const Eigen::Vector4f& p2,
		Eigen::Vector3f& p0_out,
		Eigen::Vector3f& dir_out)
	{
		Eigen::Vector3f n1 = p1.head<3>();
		Eigen::Vector3f n2 = p2.head<3>();
		float d1 = p1[3];
		float d2 = p2[3];

		Eigen::Vector3f dir = n1.cross(n2);
		float denom = dir.squaredNorm();
		if (denom < 1e-10f) return false;

		// p0 = ((d2*n1 - d1*n2) x (n1 x n2)) / |n1 x n2|^2
		Eigen::Vector3f p0 = ((d2 * n1 - d1 * n2).cross(dir)) / denom;

		p0_out = p0;
		dir_out = dir.normalized();
		return true;
	}

	inline float sqDistPointToLine(const Eigen::Vector3f& p,
		const Eigen::Vector3f& p0,
		const Eigen::Vector3f& dir_unit)
	{
		Eigen::Vector3f v = p - p0;
		Eigen::Vector3f perp = v - v.dot(dir_unit) * dir_unit;
		return perp.squaredNorm();
	}

} // namespace cc_detail

struct cloud_data
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr pc_ptr = NULL;
	int pointSize = 0;

	std::string chassisType = "";

	pcl::PointXYZ center = pcl::PointXYZ(0, 0, 0);

	int max_x = 0, min_x = 0;
	int max_y = 0, min_y = 0;
	int max_z = 0, min_z = 0;

	int len_x = 0, len_y = 0, len_z = 0;
	int len_x_min_z = 0;

	int max_x_at_max_y = 0;
	int min_x_at_max_y = 0;

	pcl::PointXYZ max_x_at_min_z = pcl::PointXYZ(0, 0, 0);
	pcl::PointXYZ min_x_at_min_z = pcl::PointXYZ(0, 0, 0);

	int midBreakPoint = 0;

	bool valid = false;
};

bool pc_passThrough(pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float minV, float maxV, std::string axis, pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult) {

	pcl::PassThrough<pcl::PointXYZ> pass;
	auto proc = false;
	auto minVal = min(minV, maxV);
	auto maxVal = max(minV, maxV);

	if (abs(maxVal - minVal) > DBL_EPSILON)
	{
		pass.setInputCloud(pcInput);
		pass.setFilterFieldName(axis);
		pass.setFilterLimits(minVal, maxVal); //min, max
		pass.filter(*pcResult);
		proc = true;
	}

	return proc;
}

bool pc_passThrough(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float minX, float maxX,
	float minY, float maxY,
	float minZ, float maxZ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	pcl::PassThrough<pcl::PointXYZ> pass;
	pcl::PointCloud<pcl::PointXYZ>::Ptr retX(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr retY(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr retZ(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr stepInput(new pcl::PointCloud<pcl::PointXYZ>);

	double minVal = 0.0;
	double maxVal = 0.0;
	bool proc = false;

	stepInput = pcInput;
	minVal = min(minX, maxX);
	maxVal = max(minX, maxX);
	if (abs(maxVal - minVal) > DBL_EPSILON)
	{
		pass.setInputCloud(stepInput);
		pass.setFilterFieldName("x");
		pass.setFilterLimits(minVal, maxVal); //min, max
		pass.filter(*retX);
		proc = true;
	}
	stepInput = proc ? retX : stepInput; proc = false;

	minVal = min(minY, maxY);
	maxVal = max(minY, maxY);
	if (abs(maxVal - minVal) > DBL_EPSILON)
	{
		pass.setInputCloud(stepInput);
		pass.setFilterFieldName("y");
		pass.setFilterLimits(minVal, maxVal); //min, max
		pass.filter(*retY);
		proc = true;
	}
	stepInput = proc ? retY : stepInput; proc = false;


	minVal = min(minZ, maxZ);
	maxVal = max(minZ, maxZ);
	if (abs(maxVal - minVal) > DBL_EPSILON)
	{
		pass.setInputCloud(stepInput);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(minVal, maxVal); //min, max
		pass.filter(*retZ);
		proc = true;
	}
	stepInput = proc ? retZ : stepInput; proc = false;

	//마지막은 없으면 없는데로 넣어줘야 한다.
	*pcResult = *stepInput;//

	return true;
}

bool pc_VoxelDown(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float leafX, float leafY, float leafZ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	if (pcInput->empty())
		return false;

	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(pcInput);
	sor.setLeafSize(leafX, leafY, leafZ);// 0.01 : 1cm
	sor.filter(*pcResult);

	return true;
}

bool pc_NoiseFilter(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	int MeanK_neighberCount,
	float StddevMulThresh,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	if (pcInput->empty())
		return false;

	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(pcInput);
	sor.setMeanK(MeanK_neighberCount); //이웃한 점의 수
	sor.setStddevMulThresh(StddevMulThresh); //노이즈 표준편차 임계값
	sor.filter(*pcResult);

	//sor.setNegative(true);
	//sor.filter(*pcOutlier);

	return true;
}

bool pc_RadiusFilter(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	float radiusSearch,
	float minNeighbors,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult
)
{
	if (pcInput->empty())
		return false;

	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	// build the filter
	outrem.setInputCloud(pcInput);
	outrem.setRadiusSearch(radiusSearch);
	outrem.setMinNeighborsInRadius(minNeighbors);
	outrem.setKeepOrganized(true);
	// apply filter
	outrem.filter(*pcResult);

	return true;
}

bool pc_CubeExtractFilter(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	Eigen::Vector3f vCubeCenter, float lengthX, float lengthY, float lengthZ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult)
{
	if (pcInput->size() <= 0)
		return false;

	float min_x = vCubeCenter.x() - lengthX;
	float max_x = vCubeCenter.x() + lengthX;

	float min_y = vCubeCenter.y() - lengthY;
	float max_y = vCubeCenter.y() + lengthY;

	float min_z = vCubeCenter.z() - lengthZ;
	float max_z = vCubeCenter.z() + lengthZ;

	return pc_passThrough(
		pcInput,
		min_x, max_x,
		min_y, max_y,
		min_z, max_z, pcResult);
}

bool pc_CountComp(const pcl::PointCloud<pcl::PointXYZ>::Ptr lhs, const pcl::PointCloud<pcl::PointXYZ>::Ptr rhs) {

	return lhs->points.size() < rhs->points.size();
}

bool pc_Clustering(
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	int sizeMin, int sizeMax, float fTolerance,
	std::list<pcl::PointCloud<pcl::PointXYZ>::Ptr>* listOut)
{
	if (pcInput->size() <= 0)
		return false;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_filtered(new pcl::PointCloud < pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_out(new pcl::PointCloud < pcl::PointXYZ>());

	cloud_filtered->resize(pcInput->size());

	pcl::copyPointCloud(*pcInput, *cloud_filtered);


	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setInputCloud(cloud_filtered);
	ec.setClusterTolerance(fTolerance); //0.02 - 2cm
	ec.setMinClusterSize(sizeMin); // 100 - 100
	ec.setMaxClusterSize(sizeMax); // 25000 - 25000
	ec.setSearchMethod(tree);
	ec.extract(cluster_indices);

	int sum = 0;
	int k = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			int idx = *pit;
			cloud_cluster->points.push_back(cloud_filtered->points[idx]); //*
		}
		//OutputDebugString("\n");
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;


		listOut->push_back(cloud_cluster);

		//sum += cloud_cluster->points.size();
	}
	listOut->sort(pc_CountComp);
	return true;
}

struct greater_pointXYZ_x
{
	inline bool operator() (const pcl::PointXYZ left, const pcl::PointXYZ right)
	{
		return (left.x > right.x);
	}
};
struct greater_pointXYZ_z
{
	inline bool operator() (const pcl::PointXYZ left, const pcl::PointXYZ right)
	{
		return (left.z > right.z);
	}
};

pcl::PointCloud<pcl::PointXYZ>::Ptr open3d_NoiseFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput,
	int MeanK_neighberCount,
	float StddevMulThresh)
{
	//2023-04-18: pcl PointCloud to open3d pointcloud convert -> filter -> revert.
	try
	{
		auto pt_cloud = open3d::geometry::PointCloud();
		auto ply_pointer = std::make_shared<open3d::geometry::PointCloud>(pt_cloud);

		auto d_points = std::vector<Eigen::Vector3d>();

		auto pc_data = pcInput->points;
		for (int i = 0; i < pc_data.size(); i++)
		{
			Eigen::Vector3d temp(pc_data.at(i).x, pc_data.at(i).y, pc_data.at(i).z);
			d_points.push_back(temp);
		}

		ply_pointer->points_ = d_points;

		//auto time_start = std::chrono::system_clock::now();
		auto nf_result = ply_pointer->RemoveStatisticalOutliers(MeanK_neighberCount, StddevMulThresh);
		//auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - time_start).count();
		//leave_a_message("open3d statistical outlier filter: " + std::to_string(dur) + "ms", std::chrono::system_clock::now());

		auto new_d_points = std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>();
		for (int i = 0; i < std::get<0>(nf_result)->points_.size(); i++)
		{
			pcl::PointXYZ temp(std::get<0>(nf_result)->points_.at(i).x(), std::get<0>(nf_result)->points_.at(i).y(), std::get<0>(nf_result)->points_.at(i).z());
			new_d_points.push_back(temp);
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr pcResult(new pcl::PointCloud<pcl::PointXYZ>);
		pcResult->points = new_d_points;

		return pcResult;
	}
	catch (...)
	{
		logMessage("Exception occurred during open3d noise filter");
		return NULL;
	}
}

auto get_center_of_cloud(const pcl::PointCloud<pcl::PointXYZ> input, int& max_x, int& min_x, int& max_y, int& min_y, int& max_z, int& min_z) -> pcl::PointXYZ
{
	//Get max min for each axis
	auto loc_max_x = -100000;
	auto loc_min_x = 100000;

	auto loc_max_y = -100000;
	auto loc_min_y = 100000;

	auto loc_max_z = -100000;
	auto loc_min_z = 100000;

	for (auto it = input.points.begin(); it < input.points.end(); ++it)
	{
		auto val_x = (*it).x;
		auto val_y = (*it).y;
		auto val_z = (*it).z;

		if (val_x > loc_max_x)
			loc_max_x = val_x;
		if (val_x < loc_min_x)
			loc_min_x = val_x;

		if (val_y > loc_max_y)
			loc_max_y = val_y;
		if (val_y < loc_min_y)
			loc_min_y = val_y;

		if (val_z > loc_max_z)
			loc_max_z = val_z;
		if (val_z < loc_min_z)
			loc_min_z = val_z;
	}
	max_x = loc_max_x;
	min_x = loc_min_x;
	max_y = loc_max_y;
	min_y = loc_min_y;
	max_z = loc_max_z;
	min_z = loc_min_z;
	return pcl::PointXYZ((max_x - min_x) / 2 + min_x, (max_y - min_y) / 2 + min_y, (max_z - min_z) / 2 + min_z);
}

auto average_x(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> float
{
	float sum = 0;
	int count = 0;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).x;
		sum += val;
	}
	return sum / input->points.size();
}
auto get_max_min_x(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int& maxX, int& minX) -> void
{
	int local_maxX = -10000;
	int local_minX = 10000;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).x;
		if (val > local_maxX) local_maxX = val;
		if (val < local_minX) local_minX = val;
	}

	maxX = local_maxX;
	minX = local_minX;
}
auto get_valid_max_x(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxX = -10000;
	int local_minX = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).x;
		if (val > local_maxX && val < 10000) local_maxX = val;
		if (val < local_minX && val > -10000) local_minX = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cx = local_maxX; cx > local_minX ; cx -= 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cx - 40, cx, "x", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_max_x = average_x(slice_);
				return (int)avg_max_x;

			}
		}
	}

	return local_maxX;
}
auto get_valid_min_x(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxX = -10000;
	int local_minX = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).x;
		if (val > local_maxX && val < 10000) local_maxX = val;
		if (val < local_minX && val > -10000) local_minX = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cx = local_minX; cx < local_maxX; cx += 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cx, cx + 40, "x", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_min_x = average_x(slice_);
				return (int)avg_min_x;

			}
		}
	}

	return local_minX;
}


auto average_y(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> float
{
	float sum = 0;
	int count = 0;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).y;
		sum += val;
	}
	return sum / input->points.size();
}
auto get_valid_min_y(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxY = -10000;
	int local_minY = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).y;
		if (val > local_maxY && val < 10000) local_maxY = val;
		if (val < local_minY && val > -10000) local_minY = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cy = local_minY; cy < local_maxY; cy += 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cy, cy + 40, "y", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_min_y = average_y(slice_);
				return (int)avg_min_y;

			}
		}
	}

	return local_minY;
}


auto get_max_min_y(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int& maxY, int& minY) -> void
{
	int local_maxY = -10000;
	int local_minY = 10000;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).y;
		if (val > local_maxY) local_maxY = val;
		if (val < local_minY) local_minY = val;
	}

	maxY = local_maxY;
	minY = local_minY;
}

auto average_z(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> float
{
	float sum = 0;
	int count = 0;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		sum += val;
	}
	return sum / input->points.size();
}
auto get_valid_max_z(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxZ = -10000;
	int local_minZ = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		if (val > local_maxZ && val < 10000) local_maxZ = val;
		if (val < local_minZ && val > -10000) local_minZ = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cz = local_maxZ; cz > local_minZ; cz -= 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cz - 40, cz, "z", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_max_z = average_z(slice_);
				return (int)avg_max_z;

			}
		}
	}

	return local_maxZ;
}
auto get_valid_min_z(pcl::PointCloud<pcl::PointXYZ>::Ptr input) -> int
{
	int local_maxZ = -10000;
	int local_minZ = 10000;

	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		if (val > local_maxZ && val < 10000) local_maxZ = val;
		if (val < local_minZ && val > -10000) local_minZ = val;
	}

	//slice through x to find first valid slice for min_z.
	for (int cz = local_minZ; cz < local_maxZ; cz += 10)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
		auto val = pc_passThrough(input, cz, cz + 40, "z", slice_);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
		if (val)
		{
			if (slice_->points.size() > 30)
			{
				//get average val.
				auto avg_min_z = average_z(slice_);
				return (int)avg_min_z;

			}
		}
	}

	//at this point, just return the local min z.
	return local_minZ;


}
auto get_max_min_z(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int& maxZ, int& minZ) -> void
{
	int local_maxZ = -10000;
	int local_minZ = 10000;
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = (*it).z;
		if (val > local_maxZ) local_maxZ = val;
		if (val < local_minZ) local_minZ = val;
	}

	maxZ = local_maxZ;
	minZ = local_minZ;
}

pcl::PointXYZ pixel_to_depth(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int x, int y)
{
	//512, 424

	//int x = 206;
	//int y = 296;
	pcl::PointXYZ ref_data(0, 0, 0);
	int index = y * 512 + x;
	if (y == 0) index = x;
	//int index = y * 640 + x;
	if (index > 217088 || index < 0)
	{
		index = 0;
		return ref_data;
	}

	try {
		//leave_a_message("mapping index = " + std::to_string(index), std::chrono::system_clock::now());
		if (input->points.size() > index) {
			ref_data = input->points[index];
		}
		return ref_data;
	}
	catch (...)
	{
		return ref_data;
	}
}

auto map_box_to_pc(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bbx detection) -> pcl::PointCloud<pcl::PointXYZ>::Ptr
{
	auto mark = detection.label;
	auto conf = detection.prob;
	auto x = detection.x;
	auto y = detection.y;
	auto w = detection.w;
	auto h = detection.h;

	int x1 = detection.x;
	int y1 = detection.y;
	int x2 = x1 + w;
	int y2 = y1 + h;

	//auto mp_start = std::chrono::system_clock::now();
	//offset for index.
	//y1 -= 50;
	//y2 -= 20;

	if (y1 < 0) y1 = 0;
	if (y2 < 0) y2 = 0;

	pcl::PointCloud<pcl::PointXYZ>::Ptr res_cloud = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
	std::vector<pcl::PointXYZ> temp_points = {};
	//res_cloud->points.resize(size_t(w * h));
	for (int j = y1; j <= y2; j++)
	{
		for (int i = x1; i <= x2; i++)
		{
			pcl::PointXYZ ref_point = pixel_to_depth(pointCloud, i, j);

			if ((ref_point.x != 0 && ref_point.y != 0 && ref_point.z != 0) ||
				((int)ref_point.x != 23 && (int)ref_point.y != 4 && (int)ref_point.z != -1))
			{
				//basic range.
				//if (ref_point.z > 1000 && ref_point.z < 3000)
				//{
				res_cloud->points.push_back(ref_point);
				//}//temp_points.push_back(ref_point);
			}
		}
	}

	//logMessage("map size: " + std::to_string(res_cloud->points.size()));

	//auto nf_start = std::chrono::system_clock::now();
	pcl::PointCloud<pcl::PointXYZ>::Ptr nf_cloud = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
	nf_cloud = open3d_NoiseFilter(res_cloud, 30, 1.0);
	//auto nf_end = std::chrono::system_clock::now();
	//logMessage("o3d noise filter: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(nf_end - nf_start).count()) + "ms");

	return nf_cloud;
}

auto map_box_to_pc_noisefiltertest(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bbx detection, int x_limit, int z_limit, std::string save_path = "") -> pcl::PointCloud<pcl::PointXYZ>::Ptr
{
	auto mark = detection.label;
	auto conf = detection.prob;
	auto x = detection.x;
	auto y = detection.y;
	auto w = detection.w;
	auto h = detection.h;

	int x1 = detection.x;
	int y1 = detection.y;
	int x2 = x1 + w;
	int y2 = y1 + h;

	//auto mp_start = std::chrono::system_clock::now();
	//offset for index.
	//y1 -= 50;
	//y2 -= 20;

	if (y1 < 0) y1 = 0;
	if (y2 < 0) y2 = 0;

	pcl::PointCloud<pcl::PointXYZ>::Ptr res_cloud(new pcl::PointCloud <pcl::PointXYZ>);
	std::vector<pcl::PointXYZ> temp_points = {};
	//res_cloud->points.resize(size_t(w * h));
	for (int j = y1; j <= y2; j++)
	{
		for (int i = x1; i <= x2; i++)
		{
			pcl::PointXYZ ref_point = pixel_to_depth(pointCloud, i, j);

			if ((ref_point.x != 0 && ref_point.y != 0 && ref_point.z != 0) ||
				((int)ref_point.x != 23 && (int)ref_point.y != 4 && (int)ref_point.z != -1))
			{
				if (x_limit > 0)
				{
					if (ref_point.x < x_limit && ref_point.z < z_limit)
					{
						{
							res_cloud->points.push_back(ref_point);
						}
					}
				}
				else if (x_limit < 0)
				{
					if (ref_point.x > x_limit && ref_point.z < z_limit)
					{
						{
							res_cloud->points.push_back(ref_point);
						}
					}
				}
				else if (x_limit == 0 && z_limit != 0)
				{
					if (ref_point.z < z_limit) res_cloud->points.push_back(ref_point);
				}
				else if (x_limit == 0 && z_limit == 0)
				{
					res_cloud->points.push_back(ref_point);
				}
			}
		}
	}
	if (save_path != "") pcl::io::savePCDFileBinaryCompressed(save_path + "_no_filter" + std::to_string(res_cloud->points.size()) + ".pcd", *res_cloud);

	//logMessage("map size: " + std::to_string(res_cloud->points.size()));
	pcl::PointCloud<pcl::PointXYZ>::Ptr nf_out(new pcl::PointCloud <pcl::PointXYZ>);
	int max_points = 0;
	int nf_index = -1;
	//nf1: 30, 0.1
	pcl::PointCloud<pcl::PointXYZ>::Ptr nf_cloud1(new pcl::PointCloud <pcl::PointXYZ>);
	//nf_cloud1 = open3d_NoiseFilter(res_cloud, 30, 1.0);
	//if (save_path != "") pcl::io::savePCDFileBinaryCompressed(save_path + "_nf30_1.0_pc_" + std::to_string(nf_cloud1->points.size()) + ".pcd", *nf_cloud1);
	if (nf_cloud1->points.size() > 0)
	{
		max_points = nf_cloud1->points.size();
		pcl::copyPointCloud(*nf_cloud1, *nf_out);
		nf_index = 1;
	}
	else
	{
		pcl::copyPointCloud(*res_cloud, *nf_out);
		nf_index = 0;
	}

	//if (save_path != "") pcl::io::savePCDFileBinaryCompressed(save_path + "_nf_out_i_" + std::to_string(nf_index) + ".pcd", *nf_out);

	return nf_out;
}

auto get_closest_xz(pcl::PointCloud<pcl::PointXYZ>::Ptr input, pcl::PointXYZ refPT) -> pcl::PointXYZ
{
	auto min = 10000;
	pcl::PointXYZ closest(0.0f, 0.0f, 0.0f);
	for (auto it = input->points.begin(); it < input->points.end(); ++it)
	{
		auto val = sqrt(pow((*it).x - refPT.x, 2) + pow((*it).z - refPT.z, 2));
		if (val < min)
		{
			min = val;
			closest = (*it);
		}
	}
	return closest;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr makePCL_PointCloud(std::vector<visionary::PointXYZ> input)
{
	try
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud <pcl::PointXYZ>);

		pointCloud->points.resize(input.size());

		std::transform(input.begin(), input.end(), pointCloud->points.begin(),
			[](visionary::PointXYZ val) { return pcl::PointXYZ(val.x * 1000, val.y * 1000, val.z * 1000); });

		return pointCloud;
	}
	catch (std::exception& ex)
	{
		logMessage("std exception occurred in pcl pointer making from T-Mini data: " + std::string(ex.what()));
		return NULL;
	}
}

//for early logging files 
pcl::PointCloud<pcl::PointXYZ>::Ptr makePCL_PointCloud(pcl::PointCloud<pcl::PointXYZ> input, bool convert = false)
{
	//auto dt_now = std::chrono::system_clock::now();
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ >);

	pointCloud->points.resize(input.size());

	if (convert)
	{
		std::transform(input.points.begin(), input.points.end(), pointCloud->points.begin(),
			[](pcl::PointXYZ val) { return pcl::PointXYZ(val.x * 1000, val.y * 1000, val.z * 1000); });
	}
	else
	{
		std::transform(input.points.begin(), input.points.end(), pointCloud->points.begin(),
			[](pcl::PointXYZ val) { return pcl::PointXYZ(val.x, val.y, val.z); });
	}
	//auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - dt_now).count();
	//std::cout << "make dur ms: " << std::to_string(dur) << std::endl;

	//pcl::io::savePCDFileBinaryCompressed("temp.pcd", *pointCloud);
	return pointCloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr init_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr pcInput, bool isLeftSide, int outX_range, int inX_range, int maxZ_range, int minZ_range, std::string object, std::string save_path = std::string(""))
{
	//1. X-Z directional limit for bigger ROI.
	int local_min_x, local_max_x;
	int local_min_z, local_max_Z;

	pcl::PointCloud<pcl::PointXYZ>::Ptr init_hole_filter_x(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr init_hole_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	local_min_z = get_valid_min_z(pcInput);
	if (isLeftSide)
	{
		//max x
		local_max_x = get_valid_max_x(pcInput);
		auto val = pc_passThrough(pcInput, local_max_x - inX_range, local_max_x + outX_range, "x", init_hole_filter_x);
		val = pc_passThrough(init_hole_filter_x, local_min_z - minZ_range, local_min_z + maxZ_range, "z", init_hole_filtered);

	}
	else
	{
		//min x
		local_min_x = get_valid_min_x(pcInput);
		auto val = pc_passThrough(pcInput, local_min_x - outX_range, local_min_x + inX_range, "x", init_hole_filter_x);
		val = pc_passThrough(init_hole_filter_x, local_min_z - minZ_range, local_min_z + maxZ_range, "z", init_hole_filtered);
	}
	if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_" + object + "_filtered.pcd", *init_hole_filtered);

	return init_hole_filtered;
}

//PointCloud :: Hole Detection
pcl::PointXYZ pc_hole_detection(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	try
	{
		pcl::PointXYZ hole_position = pcl::PointXYZ(-1, -1, -1);

		//target position is most left-lower corner point on the cornercastings.

		//main extraction region is gantry side of the cornercastings.
		//once trolley moves (-), trolley side of the cornercastings may not be visible.

		//attempt1: detect center of the hole. then use offset.
		//Use slice width of 40mm.
		//get min, max Z.
		int maxZ, minZ;
		get_max_min_z(pcInput, ref(maxZ), ref(minZ));
		int maxX, minX;
		get_max_min_x(pcInput, ref(maxX), ref(minX));

		int slice_width = 50;
		int inc = (maxZ - minZ) / slice_width;
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr mostPoints_z_slice = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
		int max_points_z = 0;
		
		//logMessage("minz: " + std::to_string(minZ) + " maxz: " + std::to_string(maxZ));

		try
		{
			for (int current_z = minZ; current_z < (maxZ - slice_width); current_z += 5)
			{
				//logMessage("current_z: " + std::to_string(current_z) + " , to z + slice: " + std::to_string(current_z + slice_width));
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
				auto val = pc_passThrough(pcInput, current_z, current_z + slice_width, "z", slice_);
				if (max_points_z < slice_->points.size())
				{
					pcl::copyPointCloud(*slice_, *mostPoints_z_slice);
					max_points_z = slice_->points.size();
					//logMessage("pointcount: " + std::to_string(max_points_z));
				}	
			}
		}
		catch (exception& ex)
		{
			logMessage(ex.what());
		}
		//logMessage("mostPoints z: " + std::to_string(mostPoints_z_slice->points.size()));
		if (mostPoints_z_slice->points.size() > 0)
		{
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_mostPointsZ_" + std::to_string(mostPoints_z_slice->points.size()) + ".pcd", *mostPoints_z_slice);

			//slice through y,
			int maxY, minY;
			get_max_min_y(mostPoints_z_slice, ref(maxY), ref(minY));
			logMessage("miny: " + std::to_string(minY) + " maxy: " + std::to_string(maxY));

			int mpz_maxZ, mpz_minZ;
			get_max_min_z(mostPoints_z_slice, ref(mpz_maxZ), ref(mpz_minZ));
			logMessage("minz: " + std::to_string(mpz_minZ) + " maxz: " + std::to_string(mpz_maxZ));

			int slice_y_width = 20;
			int y_inc = (maxY - minY) / slice_y_width;

			std::vector<int>hole_center_by_y = {};
			
			int y_index = 0;
			for (auto current_y = minY; current_y <= maxY; current_y += slice_y_width, y_index++)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr slice_y(new pcl::PointCloud<pcl::PointXYZ>);
				pc_passThrough(mostPoints_z_slice, current_y, current_y + slice_y_width, "y", slice_y);

				//slice through x,
				if (slice_y->points.size() > 0)
				{
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_mpz_slice_y_" + std::to_string(y_index) + ".pcd", *slice_y);

					int maxX, minX;
					get_max_min_x(slice_y, ref(maxX), ref(minX));
					int lenX = maxX - minX;
					//15% front and back should not be considered.
					

					int slice_x_width = 5;
					int x_inc = (maxX - minX) / slice_x_width;
					int minX_threshold = std::ceil(x_inc * 0.15);
					int maxX_threshold = std::floor(x_inc * 0.85);

					int hole_start_index = -1;
					int hole_end_index = -1;

					int current_idx = 0;
					int consecutive_count = 0;
					int consecutive_flag = false;
					for (auto current_x = minX; current_x <= maxX; current_x += 5)
					{
						pcl::PointCloud<pcl::PointXYZ>::Ptr slice_x(new pcl::PointCloud<pcl::PointXYZ>);
						pc_passThrough(slice_y, current_x, current_x + slice_x_width, "x", slice_x);

						//but needs to be consecutive.
						//look for non point index.
						if (hole_start_index == -1 && slice_x->points.size() == 0 && current_idx >= minX_threshold)
						{
							hole_start_index = current_idx;
						}
						else if (hole_start_index != -1 && slice_x->points.size() == 0 && hole_end_index == -1)
						{
							consecutive_count++;
						}
						//then check for index with points.
						else if (hole_start_index != -1 && slice_x->points.size() > 0 && current_idx <= maxX_threshold)
						{
							if (consecutive_count > 4)
							{
								//valid end of the hole.
								hole_end_index = current_idx;
								break;
							}
							else
							{
								//insufficient. reset start of the hole.
								hole_start_index = -1;
								consecutive_count = 0;
							}

						}

						current_idx++;
					}

					if (hole_start_index != -1 && hole_end_index != -1)
					{
						int offset_index = (hole_end_index - hole_start_index) / 2 + hole_start_index;
						int hole_center = minX + slice_x_width * offset_index;

						hole_center_by_y.push_back(hole_center);
					}
				}
			}
			
			//average out the center of the gap = center of the hole.
			if (hole_center_by_y.size() > 0)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr hole_center_points(new pcl::PointCloud<pcl::PointXYZ>);
				for (auto& hc_x : hole_center_by_y)
				{
					hole_center_points->points.push_back(pcl::PointXYZ(hc_x, minY, minZ));
				}
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_center_points_count_" + std::to_string(hole_center_by_y.size()) + ".pcd", *hole_center_points);

				double hole_center = static_cast<double>(std::accumulate(hole_center_by_y.begin(), hole_center_by_y.end(), 0)) / hole_center_by_y.size();

				//x: hole_center - cornercastings width / 2. = 162/2 = 81mm.
				//y: min y?
				//z: min z?
				if (isLeftSide)
				{
					hole_position.x = hole_center + 81;
				}
				else
				{
					hole_position.x = hole_center - 81;
				}

				hole_position.y = minY;
				hole_position.z = mpz_minZ;

			}
			else
			{
				logMessage("Hole Center algorithm failed.");
			}
		}
		else
		{
			logMessage("Hole has no valid z slice!");
		}



		//slice through z direction to get most points slice.
		//pcl::PointCloud<pcl::PointXYZ>::Ptr most_points_slice
		
		
		//slice through x direction to get most points slice.


		//slice through y direction on most points z slice and average out x,z position while using the lowest y position.


		return hole_position;

	}
	catch (exception& ex)
	{
		logMessage(std::string("Exception in hole detection: ") + ex.what());
	}
	catch (...)
	{
		logMessage("Unknown error from hole detection");
	}
}

pcl::PointXYZ pc_hole_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ hole_position(10000, 10000, 10000);
	try
	{
		//1. X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_hole_filter_x(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_hole_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (isLeftSide)
		{
			//max x
			local_max_x = get_valid_max_x(pcInput);
			auto val = pc_passThrough(pcInput, local_max_x - 200, local_max_x + 100, "x", init_hole_filter_x);
			val = pc_passThrough(init_hole_filter_x, local_min_z - 100, local_min_z + 200, "z", init_hole_filtered);

		}
		else
		{
			//min x
			local_min_x = get_valid_min_x(pcInput);
			auto val = pc_passThrough(pcInput, local_min_x - 100, local_min_x + 200, "x", init_hole_filter_x);
			val = pc_passThrough(init_hole_filter_x, local_min_z - 100, local_min_z + 200, "z", init_hole_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_hole_filtered.pcd", *init_hole_filtered);

		//Get Y slices :: 20 height slices.
		//Cornercastings height = usually 120mm.

		int y_slice_height = 20;

		int minY, maxY;
		get_max_min_y(init_hole_filtered, ref(maxY), ref(minY));

		std::vector<pcl::PointXYZ> target_points = {};
		std::vector<pcl::PointXYZ> target_points_rev2 = {};

		int i = 0;
		for (int cy = minY; cy < maxY; cy += y_slice_height)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_hole_filtered, cy, cy + y_slice_height, "y", slice_);
			//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val && slice_->points.size() > 50) //pc limit.
			{
				//Get most points MinX and most points MinZ Slices

				pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);

				int slice_width = 30;

				int minZ, maxZ, minX, maxX;
				get_max_min_x(slice_, ref(maxX), ref(minX));
				get_max_min_z(slice_, ref(maxZ), ref(minZ));

				for (int cx = minX; cx < maxX; cx += slice_width)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr sliceX_(new pcl::PointCloud<pcl::PointXYZ>);
					auto valX = pc_passThrough(slice_, cx, cx + slice_width, "x", sliceX_);
					if (sliceX_->points.size() > mpx_slice->points.size())
					{
						pcl::copyPointCloud(*sliceX_, *mpx_slice);
					}
				}

				for (int cz = minZ; cz < maxZ; cz += slice_width)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr sliceZ_(new pcl::PointCloud<pcl::PointXYZ>);
					auto valZ = pc_passThrough(slice_, cz, cz + slice_width, "z", sliceZ_);
					if (sliceZ_->points.size() > mpz_slice->points.size())
					{
						pcl::copyPointCloud(*sliceZ_, *mpz_slice);
					}
				}
				
				//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_i_" + std::to_string(i) + "_mpx_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
				//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_i_" + std::to_string(i) + "_mpz_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

				//from minX slice, get minZ.
				int mpx_minZ = -10000, mpx_maxZ = -10000;
				int mpx_minX = -10000, mpx_maxX = -10000;
				if (mpx_slice->points.size() > 0) 
				{
					get_max_min_x(mpx_slice, ref(mpx_maxX), ref(mpx_minX));
					get_max_min_z(mpx_slice, ref(mpx_maxZ), ref(mpx_minZ));	
				}

				//from minZ slice, get minX.
				int mpz_minX = -10000, mpz_maxX = -10000;
				int mpz_minZ = -10000, mpz_maxZ = -10000;
				if (mpz_slice->points.size() > 0) 
				{
					get_max_min_x(mpz_slice, ref(mpz_maxX), ref(mpz_minX));
					get_max_min_z(mpz_slice, ref(mpz_maxZ), ref(mpz_minZ));
				}
				 
				//then get closest "actual" point to X,Z values.
				if (isLeftSide)
				{
					if (mpx_minZ != -10000 && mpz_maxX != -10000)
					{
						pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);
						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = cy;
						
						//pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						//tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_rev2_i_" + std::to_string(i) + ".pcd", *tPt);
						target_points.push_back(targetPt);
					}
					else if (mpx_minZ == -10000 && mpz_maxX != -10000)
					{
						pcl::PointXYZ refPoint(mpz_maxX, 0, mpz_minZ);
						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = cy;

						//pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						//tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_rev2_i_" + std::to_string(i) + ".pcd", *tPt);
						target_points.push_back(targetPt);
					}
					else if (mpx_minZ != -10000 && mpz_maxX == -10000)
					{
						pcl::PointXYZ refPoint(mpx_maxX, 0, mpx_minZ);
						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = cy;

						//pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						//tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_rev2_i_" + std::to_string(i) + ".pcd", *tPt);
						target_points.push_back(targetPt);
					}
				}
				else
				{
					if (mpz_minX != -10000 && mpx_minZ != -10000)
					{
						pcl::PointXYZ refPoint(mpz_minX, 0, mpx_minZ);
						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = cy;

						//pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						//tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_rev2_i_" + std::to_string(i) + ".pcd", *tPt);
						target_points.push_back(targetPt);
					}
					else if (mpx_minX == -10000 && mpz_minX != -10000)
					{
						pcl::PointXYZ refPoint(mpz_minX, 0, mpz_minZ);
						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = cy;

						//pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						//tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_rev2_i_" + std::to_string(i) + ".pcd", *tPt);
						target_points.push_back(targetPt);
					}
					else if (mpx_minX != -10000 && mpz_minX == -10000)
					{
						pcl::PointXYZ refPoint(mpx_maxX, 0, mpx_minZ);
						//Get Closest Point.
						auto targetPt = get_closest_xz(init_hole_filtered, refPoint);
						targetPt.y = cy;

						//pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
						//tPt->points.push_back(targetPt);
						//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_rev2_i_" + std::to_string(i) + ".pcd", *tPt);
						target_points.push_back(targetPt);
					}
				}
			}
			i++;
		}

		if (save_path != std::string(""))
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);

			if (target_points.size() > 0)
			{
				for (auto& v : target_points)
				{
					tPt->points.push_back(v);
				}

				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPts_all.pcd", *tPt);
			}
		}
		//based on target points.
		//average x and z while get min y.
		float avg_x = 0.0f, avg_z = 0.0f;
		for (auto& it : target_points)
		{
			if (it.y < hole_position.y) hole_position.y = it.y;
			avg_x += it.x;
			avg_z += it.z;
		}
		avg_x = avg_x / target_points.size();
		avg_z = avg_z / target_points.size();

		hole_position.x = avg_x;
		hole_position.z = avg_z;

	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_HOLE] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_HOLE] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return hole_position;
}

// Corner definition (your semantics):
// left  = max X
// lower = min Y
// close = min Z (depth)
// Visible faces: X-normal plane and Z-normal plane
pcl::PointXYZ FindCornerCastingCorner_XZPlanes_LowerLeftClosest(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in)
{
	pcl::PointXYZ invalid;
	invalid.x = invalid.y = invalid.z = std::numeric_limits<float>::quiet_NaN();
	if (!cloud_in || cloud_in->empty()) return invalid;

	// ---- Tunables (start here) ----
	const float voxel_leaf = 0.003f;        // 3mm (2~5mm typical)
	const float ror_radius = 0.02f;         // 2cm
	const int   ror_min_neighbors = 8;

	const float ransac_dist = 0.006f;       // 6mm (tune for your distance/noise)
	const int   ransac_iters = 300;
	const float eps_angle = 10.0f * float(M_PI) / 180.0f; // 10 deg

	const float line_band = 0.010f;         // 10mm point-to-line acceptance
	const float line_band2 = line_band * line_band;

	// ---- 1) Downsample ----
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZ>);
	{
		pcl::VoxelGrid<pcl::PointXYZ> vg;
		vg.setInputCloud(cloud_in);
		vg.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
		vg.filter(*cloud_ds);
	}
	if (cloud_ds->empty()) return invalid;

	// ---- 2) Outlier removal ----
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	{
		pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
		ror.setInputCloud(cloud_ds);
		ror.setRadiusSearch(ror_radius);
		ror.setMinNeighborsInRadius(ror_min_neighbors);
		ror.filter(*cloud_f);
	}
	if (cloud_f->size() < 20) cloud_f = cloud_ds;
	if (cloud_f->empty()) return invalid;

	// ---- 3) Fit X-normal plane (side face) ----
	pcl::ModelCoefficients coeffX;
	pcl::PointIndices inliersX;
	bool okX = cc_detail::fitParallelPlane(
		cloud_f, Eigen::Vector3f::UnitX(), eps_angle, ransac_dist, ransac_iters, coeffX, inliersX);

	// ---- 4) Fit Z-normal plane (back face, depth-facing) ----
	pcl::ModelCoefficients coeffZ;
	pcl::PointIndices inliersZ;
	bool okZ = cc_detail::fitParallelPlane(
		cloud_f, Eigen::Vector3f::UnitZ(), eps_angle, ransac_dist, ransac_iters, coeffZ, inliersZ);

	// Fallback: pure robust lexicographic if plane fit fails
	auto betterFallback = [](const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
		if (a.z != b.z) return a.z < b.z;     // closest first (min Z)
		if (a.x != b.x) return a.x > b.x;     // then left (max X)
		return a.y < b.y;                     // then lower (min Y)
		};

	if (!okX || !okZ) {
		pcl::PointXYZ best = cloud_f->points[0];
		for (const auto& p : cloud_f->points) {
			if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
			if (betterFallback(p, best)) best = p;
		}
		return best;
	}

	// ---- 5) Intersection line of the two planes (should be ~ parallel to Y) ----
	Eigen::Vector4f pX(coeffX.values[0], coeffX.values[1], coeffX.values[2], coeffX.values[3]);
	Eigen::Vector4f pZ(coeffZ.values[0], coeffZ.values[1], coeffZ.values[2], coeffZ.values[3]);
	cc_detail::normalizePlane(pX);
	cc_detail::normalizePlane(pZ);

	Eigen::Vector3f p0, dir;
	if (!cc_detail::intersectTwoPlanes(pX, pZ, p0, dir)) {
		// Degenerate -> fallback
		pcl::PointXYZ best = cloud_f->points[0];
		for (const auto& p : cloud_f->points) {
			if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
			if (betterFallback(p, best)) best = p;
		}
		return best;
	}

	// ---- 6) Among points near the intersection line, pick lower=minY.
	// Tie-break with left=maxX then closest=minZ (your exact semantics).
	bool found = false;
	pcl::PointXYZ best;
	best.y = std::numeric_limits<float>::infinity();
	best.x = -std::numeric_limits<float>::infinity();
	best.z = std::numeric_limits<float>::infinity();

	for (const auto& pt : cloud_f->points) {
		if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

		Eigen::Vector3f v(pt.x, pt.y, pt.z);
		float d2 = cc_detail::sqDistPointToLine(v, p0, dir);
		if (d2 > line_band2) continue;

		// Primary: min Y (lower). Then max X (left). Then min Z (closest).
		if (!found ||
			(pt.y < best.y) ||
			(pt.y == best.y && pt.x > best.x) ||
			(pt.y == best.y && pt.x == best.x && pt.z < best.z))
		{
			best = pt;
			found = true;
		}
	}

	if (found) return best;

	// ---- 7) If no near-line points, return geometric corner from planes + minY from cloud ----
	pcl::PointXYZ minPt, maxPt;
	pcl::getMinMax3D(*cloud_f, minPt, maxPt);
	float y_target = minPt.y;

	// If dir.y is tiny, can't reliably intersect at y; just return p0 with y_target
	if (std::abs(dir.y()) < 1e-4f) {
		return pcl::PointXYZ(p0.x(), y_target, p0.z());
	}

	float t = (y_target - p0.y()) / dir.y();
	Eigen::Vector3f pg = p0 + t * dir;
	return pcl::PointXYZ(pg.x(), pg.y(), pg.z());
}


//PointCloud :: Cone Detection (XT)?
pcl::PointXYZ pc_cone_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ cone_position(10000, 10000, 10000);
	try
	{
		//X-Z directional limit for bigger ROI.
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_x(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (isLeftSide)
		{
			//max x
			local_max_x = get_valid_max_x(pcInput);
			auto val = pc_passThrough(pcInput, local_max_x - 300, local_max_x + 100, "x", init_cone_filter_x);
			val = pc_passThrough(init_cone_filter_x, local_min_z - 100, local_min_z + 300, "z", init_cone_filtered);

		}
		else
		{
			//min x
			local_min_x = get_valid_min_x(pcInput);
			auto val = pc_passThrough(pcInput, local_min_x - 100, local_min_x + 300, "x", init_cone_filter_x);
			val = pc_passThrough(init_cone_filter_x, local_min_z - 100, local_min_z + 300, "z", init_cone_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_cone_filtered.pcd", *init_cone_filtered);


		get_max_min_x(init_cone_filtered, ref(local_max_x), ref(local_min_x));
		get_max_min_z(init_cone_filtered, ref(local_max_Z), ref(local_min_z));


		//slice through y direction with slice height = 20.

		//Get most points y slice.
		//In the process of finding most points y slice, check for increase in point count then decrease in point count?

		//from most points y slice, slice through x,z for most points and look for target point.

		int y_slice_height = 30;

		int minY, maxY;
		get_max_min_y(init_cone_filtered, ref(maxY), ref(minY));

		pcl::PointCloud<pcl::PointXYZ>::Ptr mpy_slice(new pcl::PointCloud<pcl::PointXYZ>);

		int i = 0;
		int most_points_i = -1;
		for (int cy = maxY; cy > minY; cy -= y_slice_height)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_cone_filtered, cy - y_slice_height, cy, "y", slice_);
			//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > mpy_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpy_slice);
					most_points_i = i;
				}
			}
			i++;
		}

		if (most_points_i != -1)
		{
			int slice_max_y = maxY - y_slice_height * most_points_i;
			int slice_min_y = slice_max_y - y_slice_height - 15;
			slice_max_y += 15;

			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(init_cone_filtered, slice_min_y, slice_max_y, "y", slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_extended_slice_y_pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (slice_->points.size() > mpy_slice->points.size())
			{
				pcl::copyPointCloud(*slice_, *mpy_slice);
			}
		}
		
		if (mpy_slice->points.size() > 0)
		{
			int slice_width = 20;
			int slice_inc = 10;
			pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);

			int minZ, maxZ, minX, maxX;
			get_max_min_x(mpy_slice, ref(maxX), ref(minX));
			get_max_min_z(mpy_slice, ref(maxZ), ref(minZ));

			for (int cx = minX; cx < maxX; cx += slice_inc)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr sliceX_(new pcl::PointCloud<pcl::PointXYZ>);
				auto valX = pc_passThrough(mpy_slice, cx, cx + slice_width, "x", sliceX_);
				if (sliceX_->points.size() > mpx_slice->points.size())
				{
					pcl::copyPointCloud(*sliceX_, *mpx_slice);
				}
			}

			for (int cz = minZ; cz < maxZ; cz += slice_width)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr sliceZ_(new pcl::PointCloud<pcl::PointXYZ>);
				auto valZ = pc_passThrough(mpy_slice, cz, cz + slice_width, "z", sliceZ_);
				if (sliceZ_->points.size() > mpz_slice->points.size())
				{
					pcl::copyPointCloud(*sliceZ_, *mpz_slice);
				}
			}

			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_mpx_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_mpz_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

			//from minX slice, get minZ.
			int mpx_minX = -10000, mpx_maxX = -10000;
			int mpx_minZ = -10000, mpx_maxZ = -10000;
			if (mpx_slice->points.size() > 0)
			{
				get_max_min_x(mpx_slice, ref(mpx_maxX), ref(mpx_minX));
				get_max_min_z(mpx_slice, ref(mpx_maxZ), ref(mpx_minZ));
			}

			//from minZ slice, get minX.
			int mpz_minX = -10000, mpz_maxX = -10000;
			int mpz_minZ = -10000, mpz_maxZ = -10000;
			if (mpz_slice->points.size() > 0)
			{
				get_max_min_x(mpz_slice, ref(mpz_maxX), ref(mpz_minX));
				get_max_min_z(mpz_slice, ref(mpz_maxZ), ref(mpz_minZ));
			}
			//then get closest "actual" point to X,Z values.
			if (isLeftSide)
			{
				if (mpz_maxX != -10000 && mpx_minZ != -10000)
				{
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);
					
					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_maxX != -10000 && mpz_maxX == -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpx_maxX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpx.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ == -10000 && mpz_maxX != -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpz_maxX, 0, mpz_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpz.pcd", *tPt);

					cone_position = targetPt;
				}
			}
			else
			{
				if (mpz_minX != -10000 && mpx_minX != -10000)
				{
					pcl::PointXYZ refPoint(mpz_minX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minX != -10000 && mpz_minX == -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpx_minX, 0, mpx_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpx.pcd", *tPt);

					cone_position = targetPt;
				}
				else if (mpx_minZ == -10000 && mpz_minX != -10000)
				{
					//use mpx slice to get minZ and maxX.
					pcl::PointXYZ refPoint(mpz_minX, 0, mpz_minZ);

					//Get Closest Point.
					auto targetPt = get_closest_xz(mpy_slice, refPoint);

					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_targetPt_mpz.pcd", *tPt);

					cone_position = targetPt;
				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_CONE] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_CONE] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}

//PointCloud :: Landed Detection
pcl::PointXYZ pc_landed_detection_naive(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ cone_position(10000, 10000, 10000);
	try
	{
		int local_min_x, local_max_x;
		int local_min_z, local_max_Z;

		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filter_x(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr init_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		local_min_z = get_valid_min_z(pcInput);
		if (isLeftSide)
		{
			//max x
			local_max_x = get_valid_max_x(pcInput);
			auto val = pc_passThrough(pcInput, local_max_x - 300, local_max_x, "x", init_cone_filter_x);
			val = pc_passThrough(init_cone_filter_x, local_min_z, local_min_z + 300, "z", init_cone_filtered);

		}
		else
		{
			//min x
			local_min_x = get_valid_min_x(pcInput);
			auto val = pc_passThrough(pcInput, local_min_x, local_min_x + 300, "x", init_cone_filter_x);
			val = pc_passThrough(init_cone_filter_x, local_min_z, local_min_z + 300, "z", init_cone_filtered);
		}
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_init_landed_cone_filtered.pcd", *init_cone_filtered);

		//noise filter
		//3. Statistical Outlier Removal
		int SOR_K = 30;
		float SOR_STD = 1.0f;
		pcl::PointCloud<pcl::PointXYZ>::Ptr sor_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pc_NoiseFilter(init_cone_filtered, SOR_K, SOR_STD, sor_filtered);

		if (save_path != std::string(""))
			pcl::io::savePCDFileBinaryCompressed(save_path + "_init_landed_NF.pcd", *sor_filtered);

		int minZ, maxZ, minX, maxX, minY, maxY;
		get_max_min_x(sor_filtered, ref(maxX), ref(minX));
		get_max_min_z(sor_filtered, ref(maxZ), ref(minZ));
		get_max_min_y(sor_filtered, ref(maxY), ref(minY));

		
		//2. Voxel DownSample
		pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pc_VoxelDown(sor_filtered, 5.0f, 5.0f, 5.0f, voxel_filtered);

		if (save_path != std::string(""))
			pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_vx.pcd", *voxel_filtered);


		//take only top 100mm for landed detection.
		pcl::PointCloud<pcl::PointXYZ>::Ptr top_cone_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		auto val_top = pc_passThrough(voxel_filtered, maxY - 100, maxY, "y", top_cone_filtered);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_top_cone_filtered.pcd", *top_cone_filtered);

		//Then 



		//Get mpx, mpz slices
		int slice_width = 30;
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);

		if (!isLeftSide)
		{	
			int i = 0;
			for (int cx = minX; cx < maxX; cx += slice_width)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr sliceX_(new pcl::PointCloud<pcl::PointXYZ>);
				auto valX = pc_passThrough(top_cone_filtered, cx, cx + slice_width, "x", sliceX_);
				if (sliceX_->points.size() > 30)
				{
					pcl::copyPointCloud(*sliceX_, *mpx_slice);
					break;
				}

				i++;

				//if (i > 3) break;
			}
		}
		else
		{
			int i = 0;
			for (int cx = maxX; cx > minX; cx -= slice_width)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr sliceX_(new pcl::PointCloud<pcl::PointXYZ>);
				auto valX = pc_passThrough(top_cone_filtered, cx - slice_width, cx, "x", sliceX_);
				if (sliceX_->points.size() > 30)
				{
					pcl::copyPointCloud(*sliceX_, *mpx_slice);
					break;
				}

				i++;
				//if (i > 3) break;
			}
		}
		/*
		int i = 0;
		for (int cz = minZ; cz < maxZ; cz += slice_width)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr sliceZ_(new pcl::PointCloud<pcl::PointXYZ>);
			auto valZ = pc_passThrough(top_cone_filtered, cz, cz + slice_width, "z", sliceZ_);
			if (sliceZ_->points.size() > mpz_slice->points.size())
			{
				pcl::copyPointCloud(*sliceZ_, *mpz_slice);
			}
			i++;

			if (i > 3) break;
		}
		*/
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_mpx_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
		//if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_mpz_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

		//from minX slice, get minZ.
		int mpx_minX = -10000, mpx_maxX = -10000;
		int mpx_minZ = -10000, mpx_maxZ = -10000;
		if (mpx_slice->points.size() > 0)
		{
			get_max_min_x(mpx_slice, ref(mpx_maxX), ref(mpx_minX));
			get_max_min_z(mpx_slice, ref(mpx_maxZ), ref(mpx_minZ));
		}
		
		//from minZ slice, get minX.
		int mpz_minX = -10000, mpz_maxX = -10000;
		int mpz_minZ = -10000, mpz_maxZ = -10000;
		/*
		if (mpz_slice->points.size() > 0)
		{
			get_max_min_x(mpz_slice, ref(mpz_maxX), ref(mpz_minX));
			get_max_min_z(mpz_slice, ref(mpz_maxZ), ref(mpz_minZ));
		}
		*/
		//then get closest "actual" point to X,Z values.
		if (isLeftSide)
		{
			if (mpz_maxX != -10000 && mpx_minZ != -10000)
			{
				pcl::PointXYZ refPoint(mpz_maxX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt.pcd", *tPt);

				cone_position = targetPt;
			}
			else if (mpx_maxX != -10000 && mpz_minZ == -10000)
			{
				//use mpx slice to get minZ and maxX.
				pcl::PointXYZ refPoint(mpx_maxX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt_mpx.pcd", *tPt);

				cone_position = targetPt;
			}
			else if (mpx_maxX == -10000 && mpz_minZ != -10000)
			{
				//use mpx slice to get minZ and maxX.
				pcl::PointXYZ refPoint(mpz_maxX, 0, mpz_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt_mpz.pcd", *tPt);

				cone_position = targetPt;
			}
		}
		else
		{
			if (mpz_minX != -10000 && mpx_minZ != -10000)
			{
				pcl::PointXYZ refPoint(mpz_minX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt.pcd", *tPt);

				cone_position = targetPt;
			}
			else if (mpx_minX != -10000 && mpz_minZ == -10000)
			{
				//use mpx slice to get minZ and maxX.
				pcl::PointXYZ refPoint(mpx_minX, 0, mpx_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt_mpx.pcd", *tPt);

				cone_position = targetPt;
			}
			else if (mpx_minX == -10000 && mpz_minZ != -10000)
			{
				//use mpx slice to get minZ and maxX.
				pcl::PointXYZ refPoint(mpz_minX, 0, mpz_minZ);

				//Get Closest Point.
				auto targetPt = get_closest_xz(init_cone_filtered, refPoint);
				targetPt.y = maxY;

				pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
				tPt->points.push_back(targetPt);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_targetPt_mpz.pcd", *tPt);

				cone_position = targetPt;
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_LANDED] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_LANDED] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}

//PointCloud :: Guide Detection

cv::Mat drawOnImage(cv::Mat input, std::vector<rectangle_info> det_results)
{
	cv::Mat result_image = input.clone();

	for (auto& it : det_results)
	{
		bbx box;
		box.label = it.ClassInfo;

		box.x = (int)it.X;
		box.y = (int)it.Y;

		box.center_x = (int)it.Center_X;
		box.center_y = (int)it.Center_Y;

		box.w = (int)it.W;
		box.h = (int)it.H;

		box.prob = (int)it.Prob;

		cv::Scalar clr(0, 0, 255);
		if (box.label == 1) clr = cv::Scalar(0, 255, 0);
		else if (box.label == 2) clr = cv::Scalar(255, 0, 0);
		else if (box.label == 3) clr = cv::Scalar(128, 128, 0);

		cv::rectangle(result_image, cv::Rect(box.x, box.y, box.w, box.h), clr);

		//put-text
		std::string label = std::to_string(box.label) + "," + std::to_string(box.prob) + "%";
		cv::putText(result_image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
	}

	return result_image;
}
cv::Mat drawOnImage(cv::Mat input, bbx boxInfo, cv::Scalar clr)
{
	cv::Mat result_image = input.clone();

	cv::rectangle(result_image, cv::Rect(boxInfo.x, boxInfo.y, boxInfo.w, boxInfo.h), clr);
	//put-text
	std::string label = std::to_string(boxInfo.label) + "," + std::to_string(boxInfo.prob) + "%";
	cv::putText(result_image, label, cv::Point(boxInfo.x, boxInfo.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);

	return result_image;
}
cv::Mat drawOnImage(cv::Mat input, std::vector<Detection> det_results)
{
	cv::Mat result_image = input.clone();

	for (auto& it : det_results)
	{
		bbx box;
		box.label = it.classId;

		box.x = (int)it.box.x;
		box.y = (int)it.box.y;

		box.center_x = (int)(it.box.x + it.box.width / 2);
		box.center_y = (int)(it.box.y + it.box.height / 2);

		box.w = (int)it.box.width;
		box.h = (int)it.box.height;

		box.prob = (int)(it.conf * 100);

		cv::Scalar clr(0, 0, 255);
		if (box.label == 1) clr = cv::Scalar(0, 255, 0);
		else if (box.label == 2) clr = cv::Scalar(255, 0, 0);
		else if (box.label == 3) clr = cv::Scalar(128, 128, 0);

		cv::rectangle(result_image, cv::Rect(box.x, box.y, box.w, box.h), clr);

		//put-text
		std::string label = std::to_string(box.label) + "," + std::to_string(box.prob) + "%";
		cv::putText(result_image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, clr, 2);
	}

	return result_image;
}
bool save_to_drive(bool isJobLog, cv::Mat oImage, cv::Mat resImage, pcl::PointCloud<pcl::PointXYZ> pointCloud, std::string msg, std::chrono::system_clock::time_point frameTime)
{
	try {
		if (isJobLog)
		{
			auto SaveOImageDir = job_result_folder_name + "/Image/original";
			createDirectory_ifexists(SaveOImageDir);
			auto SaveResImageDir = job_result_folder_name + "/Image/result";
			createDirectory_ifexists(SaveResImageDir);
			auto SaveDepthDir = job_result_folder_name + "/Depth";

			auto timeNow = time_as_name(frameTime);

			//msg: {TWL/TWUL}_{LANDED/LANDOFF}

			std::string pos = "RL";
			if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

			if (!oImage.empty()) cv::imwrite(SaveOImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_oImage.jpg", oImage);
			if (!resImage.empty()) cv::imwrite(SaveResImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_resImage.jpg", resImage);

			std::string plyFilePath = SaveDepthDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_Depth.pcd";
			if (pointCloud.points.size() > 0) pcl::io::savePCDFileBinaryCompressed(plyFilePath.c_str(), pointCloud);
		}
		else
		{
			auto SaveImageDir = saveDirName + "/Image";
			auto SaveDepthDir = saveDirName + "/Depth";

			auto timeNow = time_as_name(frameTime);

			std::string pos = "RL";
			if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

			if (!oImage.empty()) cv::imwrite(SaveImageDir + "/" + timeNow + "_TMini_" + pos + "_Image.jpg", oImage);
			//-----------------------------------------------
			// Write point cloud to PLY
			std::string plyFilePath = SaveDepthDir + "/" + timeNow + "_TMini_" + pos + "_Depth.pcd";
			if (pointCloud.points.size() > 0) pcl::io::savePCDFileBinaryCompressed(plyFilePath.c_str(), pointCloud);
			//std::printf("Writing frame to %s\n", plyFilePath);
			//PointCloudPlyWriter::WriteFormatPLY(plyFilePath.c_str(), pointCloud, intensityMap, true);
			//std::printf("Finished writing frame to %s\n", plyFilePath);
		}

		return true;
	}
	catch (...)
	{
		logMessage("[save_to_drive] Exception occurred.");
		return false;
	}
}
bool save_to_drive_optimized(bool isJobLog, cv::Mat oImage, cv::Mat resImage, pcl::PointCloud<pcl::PointXYZ> pointCloud, std::vector<uint16_t> distMap, std::string msg, std::chrono::system_clock::time_point frameTime)
{
	try {
		if (isJobLog)
		{
			auto SaveOImageDir = job_result_folder_name + "/Image/original";
			createDirectory_ifexists(SaveOImageDir);
			auto SaveResImageDir = job_result_folder_name + "/Image/result";
			createDirectory_ifexists(SaveResImageDir);
			auto SaveDepthDir = job_result_folder_name + "/Depth";
			createDirectory_ifexists(SaveDepthDir);
			//auto SaveDistDir = job_result_folder_name + "/DistMap";
			//createDirectory_ifexists(SaveDistDir);

			auto timeNow = time_as_name(frameTime);

			//msg: {TWL/TWUL}_{LANDED/LANDOFF}

			std::string pos = "RL";
			if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

			if (!oImage.empty()) cv::imwrite(SaveOImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_oImage.jpg", oImage);
			if (!resImage.empty()) cv::imwrite(SaveResImageDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_resImage.jpg", resImage);
			

			std::string plyFilePath = SaveDepthDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_Depth.pcd";
			//if (pointCloud.points.size() > 0) pcl::io::savePCDFileBinaryCompressed(plyFilePath.c_str(), pointCloud);
			if (pointCloud.points.size() > 0) pcl::io::savePCDFileBinaryCompressed(plyFilePath.c_str(), pointCloud);
			//if (pointCloud.points.size() > 0) save_cloud_pcd_compressed_zstd(plyFilePath.c_str(), pointCloud);
			//if (!distMap.empty())
			//{
			//	save_u16_zstd(SaveDistDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_distMap", distMap);
			//}
		}
		else
		{
			auto SaveImageDir = saveDirName + "/Image";
			auto SaveDepthDir = saveDirName + "/Depth";
			//auto SaveDistDir = saveDirName + "/DistMap";

			auto timeNow = time_as_name(frameTime);

			std::string pos = "RL";
			if (CURRENT_SENSOR_POSITION == "REAR_RIGHT") pos = "RR";

			if (!oImage.empty()) cv::imwrite(SaveImageDir + "/" + timeNow + "_TMini_" + pos + "_Image.jpg", oImage);
			//-----------------------------------------------
			// Write point cloud to PLY
			std::string plyFilePath = SaveDepthDir + "/" + timeNow + "_TMini_" + pos + "_Depth.pcd";
			if (pointCloud.points.size() > 0)
			{
				pcl::io::savePCDFileBinaryCompressed(plyFilePath.c_str(), pointCloud);
				//save_cloud_pcd_binary_zstd(plyFilePath.c_str(), pointCloud);
			}
			//std::printf("Writing frame to %s\n", plyFilePath);
			//PointCloudPlyWriter::WriteFormatPLY(plyFilePath.c_str(), pointCloud, intensityMap, true);
			//std::printf("Finished writing frame to %s\n", plyFilePath);
			//if (!distMap.empty())
			//{
			//	save_u16_zstd(SaveDistDir + "/" + timeNow + "_" + msg + "_TMini_" + pos + "_distMap.zstd", distMap);
			//}
		}

		return true;
	}
	catch (...)
	{
		logMessage("[save_to_drive] Exception occurred.");
		return false;
	}
}
void data_save_thread()
{
	logMessage("Data Logging Thread Activated");
	try
	{
		while (logging_running.load())
		{
			//Safe blocking op.
			std::unique_lock<std::mutex> lock(mutex_logging);
			bool res = cond_logging.wait_for(lock,
				std::chrono::seconds(3600),
				[]() { return saveFlag; });

			//purposely woken up.
			int proc_count = 5;
			if (res)
			{
				while (enable_logging || tsq.GetQueueLen() > 0)
				{
					try
					{
						if (tsq.GetQueueLen() > 0)
						{
							//in blocked state until item is placed in queue.
							auto dataTup = tsq.pop();

							cv::Mat oImage; cv::Mat resImage; pcl::PointCloud<pcl::PointXYZ> pointCloud;
							std::vector<uint16_t> distMap;
							bool isJobLog = false; std::string msg;

							oImage = std::get<0>(dataTup);
							resImage = std::get<1>(dataTup);
							pointCloud = std::get<2>(dataTup);
							distMap = std::get<3>(dataTup);
							isJobLog = std::get<4>(dataTup);
							msg = std::get<5>(dataTup);
							auto frameTime = std::get<6>(dataTup);

							//bool save_res = save_to_drive(isJobLog, oImage, resImage, pointCloud, msg, frameTime);
							bool save_res = save_to_drive_optimized(isJobLog, oImage, resImage, pointCloud, distMap, msg, frameTime);
							if (!save_res) logMessage("Failed to save data to drive!");
							else logMessage("[Data-Save-Thread] Data saved successfully for " + msg + " at " + time_as_name(frameTime));
							//do not hug the process.
							std::this_thread::sleep_for(std::chrono::milliseconds(25));
						}
					}
					catch (std::exception& ex)
					{
						logMessage("[Data-Save-Thread] " + std::string(ex.what()));
						std::this_thread::sleep_for(std::chrono::milliseconds(10));
					}
					catch (...)
					{
						logMessage("[Data-Save-Thread] Unknown Exception!");
						std::this_thread::sleep_for(std::chrono::milliseconds(10));
					}
				}
			}
		}

		logMessage("Data Logging Thread Deactivated");
	}
	catch (std::exception& ex)
	{
		logMessage("[Data-Save-Thread] " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("[Data-Save-Thread] Unknown Exception!");
	}	
}

auto tmini_setup() -> bool
{
	bool status = false;
	try
	{
		using namespace visionary;

		//Get IP from INI
		std::string deviceIpAddr = current_lane_ip;
		unsigned short deviceBlobCtrlPort = 2114;
		unsigned cnt = 100u;

		controlConnected = false;
		dataConnected = false;

		logMessage("Connecting to sensor : " + current_lane_ip);

		//-----------------------------------------------
		// Connect to devices data stream 
		if (!dataStream.open(deviceIpAddr, htons(deviceBlobCtrlPort)))
		{
			logMessage("Failed to open data stream connection to device.");
			return false;   // connection failed
		}

		logMessage("Data Stream opened.");
		dataConnected = true;

		//-----------------------------------------------
		// Connect to devices control channel
		if (!visionaryControl.open(VisionaryControl::ProtocolType::COLA_2, deviceIpAddr, 5000/*ms*/))
		{
			logMessage("Failed to open control connection to device.");
			return false;   // connection failed
		}

		logMessage("Control stream opened.");
		controlConnected = true;

		//-----------------------------------------------
		// read Device Ident
		logMessage("DeviceIdent: " + visionaryControl.getDeviceIdent());
		if (!visionaryControl.logout())
		{
			logMessage("Failed to logout");
		}
		status = true;
	}
	catch (exception ex)
	{
		return false;
	}
	catch (...)
	{
		return false;
	}

	return status;
}
void thread_tmini_control()
{
	//server that controls the connection to specific "ip" address of t mini depending on lane number.
	try
	{
		while (tmini_ctrl_running.load())
		{

			std::unique_lock<std::mutex> lock(mutex_tmini_ctrl);
			bool res = cond_tmini_ctrl.wait_for(lock,
				std::chrono::seconds(3600),
				[]() { return tmini_ctrl_flag; });

			if (tmini_ctrl_flag)
			{
				sensor_last_attempted_time = std::chrono::system_clock::now();
				logMessage("TMini Setup Start!");
				auto status = tmini_setup();
				if (status)
				{
					sensor_last_connected_time = std::chrono::system_clock::now();
					sensor_connected = true;
					sensor_fault = false;

					tmini_ctrl_flag = false; //dont enter this loop again.

					//allow streaming to start.
					tmini_data_stream_flag = true;
					cond_tmini_data_stream.notify_one();

					logMessage("TMini Setup Success!");
				}
				else
				{
					
					sensor_connected = false;
					sensor_fault = true;

					logMessage("TMini Setup Failed!");
				}
			}
		}

		logMessage("Terminating tmini control thread");

	}
	catch (std::exception& ex)
	{
		logMessage("[TMini-Control] " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("[TMini-Control] Unknown Exception!");
	}
}

pcl::PointCloud<pcl::PointXYZ>::Ptr demo_ply(new pcl::PointCloud<pcl::PointXYZ>);
cv::Mat demo_img;

void printMatType(const cv::Mat& mat)
{
	int type = mat.type(); // Get the type of the Mat

	// Get the number of channels
	int channels = mat.channels();

	// Print the type based on the channels and depth
	std::string outMsg = "Type: ";

	switch (CV_MAT_DEPTH(type)) {
	case CV_8U: outMsg += "CV_8U"; break;
	case CV_8S: outMsg += "CV_8S"; break;
	case CV_16U: outMsg += "CV_16U"; break;
	case CV_16S: outMsg += "CV_16S"; break;
	case CV_32S: outMsg += "CV_32S"; break;
	case CV_32F: outMsg += "CV_32F"; break;
	case CV_64F: outMsg += "CV_64F"; break;
	default: outMsg += "Unknown depth"; break;
	}

	// Print the number of channels
	if (channels == 1) {
		outMsg += "C1";  // Single channel
	}
	else if (channels == 3) {
		outMsg += "C3";  // Three channels (e.g., RGB)
	}
	else if (channels == 4) {
		outMsg += "C4";  // Four channels (e.g., RGBA)
	}
	else {
		outMsg += "C" + channels;  // General case
	}

	logMessage(outMsg);

}

cv::VideoCapture tryOpenStream(const std::string& url, bool& success) {
	cv::VideoCapture cap(url, cv::CAP_FFMPEG);
	success = cap.isOpened();
	return cap;
}

auto tmini_data_stream() -> void
{
	try
	{
		if (visionaryControl.stopAcquisition())
			logMessage("Data Acquisition Stopped");

		if (tmini_data_stream_flag) logMessage("Data Streaming Ready & Starting!");
		while (tmini_data_stream_flag)
		{
			auto waitDur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_frame_get_time).count();
			if (waitDur_ms > 100)
			{
				blnGetNewFrame = false;
				bool stepComplete = visionaryControl.stepAcquisition();
				if (stepComplete)
				{
					if (dataStream.getNextFrame())
					{
						blnGetNewFrame = true;
						last_frame_get_time = std::chrono::system_clock::now();

						//no wait.	
						bool isFrameGrabbed = false;	
						
						//std::printf("Frame received through step called, frame #%d, timestamp: %u \n", pDataHandler->getFrameNum(), pDataHandler->getTimestampMS());					
						{
							//-----------------------------------------------
							// Convert data to a point cloud
							std::vector<PointXYZ> pointCloud;
							pDataHandler->generatePointCloud(pointCloud);
							pDataHandler->transformPointCloud(pointCloud);

							auto distMap = pDataHandler->getDistanceMap();
							//logMessage("DistMap: " + std::to_string(distMap.size()));

							auto intensityMap = pDataHandler->getIntensityMap();

							auto iW = pDataHandler->getWidth();
							auto iH = pDataHandler->getHeight();
							auto gImg = cv::Mat(pDataHandler->getHeight(), pDataHandler->getWidth(), CV_16UC1, intensityMap.data());
							cv::Mat im3; // I want im3 to be the CV_16UC1 of im2
							gImg.convertTo(im3, CV_8UC1);

							cv::Mat grayToClr;
							cv::cvtColor(im3, grayToClr, cv::COLOR_GRAY2BGR);

							auto pclPointCloud = makePCL_PointCloud(pointCloud);
							if (pclPointCloud == NULL)
							{
								logMessage("[TMini-Data-Stream] Failed to create PCL Point Cloud from data handler. -- nullptr!");
								continue;
							}
							
							/*
							isFrameGrabbed = camRGB_cap.read(camRGB_img);
							if (isFrameGrabbed) logMessage("RGB Frame is properly ready");
							else logMessage("RGB Frame stream needs to delay");
							*/
							if (MODE_DEBUG)
							{
								//replace image with another image.
								logMessage("Current index: " + std::to_string(DEBUG_CURRENT_INDEX) + " Max index: " + std::to_string(DEBUG_MAX_INDEX));
								auto imgFile = cv::imread(DEBUG_IMG_FILES.at(DEBUG_CURRENT_INDEX), 0);
								auto plyFile = DEBUG_PLY_FILES.at(DEBUG_CURRENT_INDEX);

								pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
								if (pcl::io::loadPLYFile<pcl::PointXYZ>(plyFile, *pointCloud) == -1)
								{
									std::cout << std::string("Couldn't read file by PLY ") << std::endl;
									//return false;
								}

								imgFile.convertTo(im3, CV_8UC1);
								//printMatType(imgFile);
								//printMatType(im3);
								auto pclPointCloud_temp = makePCL_PointCloud(*pointCloud, DEBUG_CONVERT_PCL_RANGE);
								pcl::copyPointCloud(*pclPointCloud_temp, *pclPointCloud);

								if (DEBUG_CURRENT_INDEX < DEBUG_MAX_INDEX - 1) DEBUG_CURRENT_INDEX++;
								else DEBUG_CURRENT_INDEX = 0;
							}

							if (true)
							{
								logMessage("Data stack updating stack with new data...");
								auto res = dataStack.Update_Stack(grayToClr, *pclPointCloud, distMap, last_frame_get_time);
								if (res) logMessage("Data Stack Updated Successfully!");
								else logMessage("Data Stack Update Failed!");

								if (enable_stream)
								{
									cv::imshow(CURRENT_SENSOR_POSITION + " T Mini", grayToClr);
									cv::waitKey(1);
								}
								else
								{
									cv::destroyAllWindows();
								}
							}
						}
					}
					else
					{
						logMessage("Data Acqusition Failed!");
						std::this_thread::sleep_for(std::chrono::milliseconds(30));
					}
				}
				else
				{
					logMessage("Data Step Failed!");
					std::this_thread::sleep_for(std::chrono::milliseconds(30));
				}
			}
			else
			{
				if (waitDur_ms > 50)
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
				}
				else
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(50 - waitDur_ms));
				}
			}
		}

		cv::destroyAllWindows();
		visionaryControl.close();
		dataStream.close();

		logMessage("[TMini-Data-Stream] Data Stream Closed!");
	}
	catch (std::exception& ex)
	{
		logMessage("[TMini-Data-Stream] " + std::string(ex.what()));
		cv::destroyAllWindows();
		visionaryControl.close();
		dataStream.close();
	}
	catch (...)
	{
		logMessage("[TMini-Data-Stream] Unknown Exception!");
		cv::destroyAllWindows();
		visionaryControl.close();
		dataStream.close();
	}
}
void thread_tmini_data_stream()
{
	try
	{
		while (tmini_data_stream_running.load())
		{
			std::unique_lock<std::mutex> lock(mutex_tmini_data_stream);
			bool res = cond_tmini_data_stream.wait_for(lock,
				std::chrono::seconds(3600),
				[]() { return tmini_data_stream_flag; });

			if (tmini_data_stream_flag)
			{
				proc_running.store(true);
				tmini_data_stream();
		
				//end sensor connection.
				//visionaryControl.close();
				//dataStream.close();

				//deflag sensor related bits.
				sensor_connected = false;
				sensor_fault = false;

				//dataStack.Clear_Stack();
				//if (ENABLE_CAM_RGB) dataStackRGB.Clear_Stack();

				logMessage("Sensor Disconnected!");
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[TMini-Data-Stream] " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("[TMini-Data-Stream] Unknown Exception!");
	}
}

//modular functions
//Inference.
/*
bool onnx_inference(VA& detector, std::string SENSOR_POSITION, cv::Mat image, cv::Mat& res_image, std::vector<rectangle_info>& det_results, std::vector<std::vector<bbx>>& det_sorted_objects, int& det_count, std::string save_path = "", bool jobLog = false)
{
	try
	{
		bool status_return = true;
		if (image.empty())
		{
			logMessage("[ONNX-Inference] Image is empty!");
			return false;
		}
		det_sorted_objects = std::vector<std::vector<bbx>>(4, std::vector<bbx>(0));

		//Image Inference
		det_results = {};
		detector.onnx_inference(image, ref(det_results));
		
		det_count = 0;
		for (auto& it : det_results)
		{
			bbx box;
			box.label = it.ClassInfo;

			box.x = (int)it.X;
			box.y = (int)it.Y;

			box.center_x = (int)it.Center_X;
			box.center_y = (int)it.Center_Y;

			box.w = (int)it.W;
			box.h = (int)it.H;

			box.prob = (int)it.Prob;

			if (box.prob < PROB_LIMIT)
			{
				logMessage("Get rid of this object: " + std::to_string(box.label) + " at prob: " + std::to_string(box.prob));
				continue;
			}

			//Choose side depending on job lane.
			if (SENSOR_POSITION == "REAR_LEFT")
			{
				if (box.center_x < 256)
				{
					logMessage("Get rid of this object RL");
					continue;
				}
			}
			else if (SENSOR_POSITION == "REAR_RIGHT")
			{
				if (box.center_x > 256)
				{
					logMessage("Get rid of this object RR");
					continue;
				}
			}

			det_count++;
			det_sorted_objects.at(box.label).push_back(box);	

			if (jobLog) jobLogMessage("Det Object: " + std::to_string(box.label) + "," + std::to_string(box.x) + "," + std::to_string(box.y) + "," + std::to_string(box.w) + "," + std::to_string(box.h) + "," + std::to_string(box.prob));	
		}
		if (save_path != std::string(""))
		{
			res_image = drawOnImage(image, det_results);
			//cv::imwrite(save_path + "_result.jpg", res_img);
		}

		if (det_count == 0) status_return = false;

	}
	catch (std::exception& ex)
	{
		logMessage("[ONNX-Inference] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[ONNX-Inference] Unknown Exception!");
		return false;
	}
}
*/
bool yolo_inference(YOLO11Detector& detector, std::string SENSOR_POSITION, cv::Mat image, cv::Mat& res_image, std::vector<rectangle_info>& det_results, std::vector<std::vector<bbx>>& det_sorted_objects, int& det_count, std::string save_path = "", bool jobLog = false)
{
	try
	{
		bool status_return = true;
		if (image.empty())
		{
			logMessage("[ONNX-Inference] Image is empty!");
			return false;
		}
		det_sorted_objects = std::vector<std::vector<bbx>>(4, std::vector<bbx>(0));

		//Image Inference
		det_results = {};

		// Perform detection with the updated thresholds
		std::vector<Detection> detections = detector.detect(image, confThreshold, iouThreshold);

		det_count = 0;
		for (auto& it : detections)
		{
			if (it.conf * 100 < PROB_LIMIT)
			{
				//logMessage("Get rid of this object: " + std::to_string(it.classId) + " at prob: " + std::to_string(it.conf * 100));
				continue;
			}

			int cx = it.box.x + it.box.width / 2;
			//Choose side depending on job lane.
			if (SENSOR_POSITION == "REAR_LEFT")
			{
				if (cx < 256)
				{
					//logMessage("Get rid of this object RL");
					continue;
				}
			}
			else if (SENSOR_POSITION == "REAR_RIGHT")
			{
				if (cx > 256)
				{
					//logMessage("Get rid of this object RR");
					continue;
				}
			}

			bbx box;
			box.label = it.classId;

			box.x = (int)it.box.x;
			box.y = (int)it.box.y;

			box.center_x = (int)(it.box.x + it.box.width / 2);
			box.center_y = (int)(it.box.y + it.box.height / 2);

			box.w = (int)it.box.width;
			box.h = (int)it.box.height;

			box.prob = (int)(it.conf * 100);

			det_count++;
			det_sorted_objects.at(box.label).push_back(box);

			if (jobLog) jobLogMessage("Det Object: " + std::to_string(box.label) + "," + std::to_string(box.x) + "," + std::to_string(box.y) + "," + std::to_string(box.w) + "," + std::to_string(box.h) + "," + std::to_string(box.prob));
		}
		
		if (save_path != std::string(""))
		{
			res_image = drawOnImage(image, detections);
			//cv::imwrite(save_path + "_result.jpg", res_img);
		}

		if (det_count == 0) status_return = false;

	}
	catch (std::exception& ex)
	{
		logMessage("[ONNX-Inference] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[ONNX-Inference] Unknown Exception!");
		return false;
	}
}

std::string chassis_type_selection_VA(std::vector<std::vector<bbx>>& det_sorted_objects, bool jobLog = false)
{
	std::string chassisType = "Unknown";
	try
	{
		if ((det_sorted_objects[1].size() > 0 || det_sorted_objects[2].size() > 0) && det_sorted_objects[3].size() == 0)
		{
			logMessage("Chassis Type: XT");
			if (jobLog) jobLogMessage("Chassis Type: XT");
			chassisType = "XT";
		}
		else if ((det_sorted_objects[1].size() == 0 && det_sorted_objects[2].size() == 0) && det_sorted_objects[3].size() > 0)
		{
			logMessage("Chassis Type: CST");
			if (jobLog) jobLogMessage("Chassis Type: CST");
			chassisType = "CST";
		}
		else if ((det_sorted_objects[1].size() > 0 || det_sorted_objects[2].size() > 0) && det_sorted_objects[3].size() > 0)
		{
			if (det_sorted_objects[1].size() > 0)
			{
				if (det_sorted_objects[1][0].prob >= det_sorted_objects[3][0].prob)
				{
					logMessage("Chassis Type: XT by prob");
					if (jobLog) jobLogMessage("Chassis Type: XT by prob");
					chassisType = "XT";
				}
				else
				{
					logMessage("Chassis Type: CST by prob");
					if (jobLog) jobLogMessage("Chassis Type: CST by prob");
					chassisType = "CST";
				}
			}
			else
			{
				if (det_sorted_objects[2][0].prob >= det_sorted_objects[3][0].prob)
				{
					logMessage("Chassis Type: XT by prob");
					if (jobLog) jobLogMessage("Chassis Type: XT by prob");
					chassisType = "XT";
				}
				else
				{
					logMessage("Chassis Type: CST by prob");
					if (jobLog) jobLogMessage("Chassis Type: CST by prob");
					chassisType = "CST";
				}
			}
		}
		else
		{
			logMessage("Chassis Type: Unknown");
			if (jobLog) jobLogMessage("Chassis Type: Unknown");
		}

		return chassisType;
	}
	catch (std::exception& ex)
	{
		logMessage("[Chassis-Type-Selection-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[Chassis-Type-Selection-VA] " + std::string(ex.what()));
		return chassisType;
	}
	catch (...)
	{
		logMessage("[Chassis-Type-Selection-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[Chassis-Type-Selection-VA] Unknown Exception!");
		return chassisType;
	}
}
bool Target_Selections_VA(std::string SENSOR_POSITION, int sprd_size, std::vector<std::vector<bbx>> det_sorted_objects, bbx& target_hole, bbx& target_cone, bbx& target_guide, bool& detected_hole, bool jobLog = false)
{
	try
	{
		//Target cornercastings selection -- Get Highest Prob.
		if (det_sorted_objects[0].size() > 0)
		{
			if (sprd_size == 45 && det_sorted_objects[0].size() > 1)
			{
				std::vector<bbx> detected_holes = det_sorted_objects[0];
				//ascending order.
				std::sort(detected_holes.begin(), detected_holes.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});

				if (SENSOR_POSITION == "REAR_LEFT")target_hole = detected_holes[0];	
				else target_hole = detected_holes[detected_holes.size() - 1];				
			}
			else
			{
				//take highest prob. -- this happens to be the case if 45ft job but single hole is detected.
				target_hole = det_sorted_objects[0][0];
			}
			detected_hole = true;
		}

		// Combine vectors using std::copy
		std::vector<bbx> detected_cones;
		detected_cones.reserve(det_sorted_objects[1].size() + det_sorted_objects[2].size());  // Reserve space to avoid multiple allocations
		detected_cones.insert(detected_cones.end(), det_sorted_objects[1].begin(), det_sorted_objects[1].end());  // Insert first vector
		std::copy(det_sorted_objects[2].begin(), det_sorted_objects[2].end(), std::back_inserter(detected_cones));  // Copy second vector

		if (detected_cones.size() > 0)
		{
			if (detected_cones.size() > 1)
			{
				// Sort the vector by center_x (ascending order)
				std::sort(detected_cones.begin(), detected_cones.end(), [](const bbx& a, const bbx& b) {
					return a.center_x < b.center_x;  // Comparison based on the 'center_x' field
					});
			}

			if (SENSOR_POSITION == "REAR_LEFT") target_cone = detected_cones[detected_cones.size() - 1];
			else if (SENSOR_POSITION == "REAR_RIGHT") target_cone = detected_cones[0];
		}

		//Target Guide -- use highest prob for now.
		if (det_sorted_objects[3].size() > 0) target_guide = det_sorted_objects[3][0];

		if (target_hole.prob > 0)
		{
			bDetected_hole = true;
			logMessage("Detected Hole at: " + std::to_string(target_hole.label) + "," + std::to_string(target_hole.center_x) + "," + std::to_string(target_hole.center_y) + " with prob: " + std::to_string(target_hole.prob));
			if (jobLog) jobLogMessage("Detected Hole at: " + std::to_string(target_hole.label) + "," + std::to_string(target_hole.center_x) + "," + std::to_string(target_hole.center_y) + " with prob: " + std::to_string(target_hole.prob));
		}
		else bDetected_hole = false;
		if (target_cone.prob > 0)
		{
			bDetected_cone = true;
			logMessage("Detected Cone at: " + std::to_string(target_cone.label) + "," + std::to_string(target_cone.center_x) + "," + std::to_string(target_cone.center_y) + " with prob: " + std::to_string(target_cone.prob));
			if (jobLog) jobLogMessage("Detected Cone at: " + std::to_string(target_cone.label) + "," + std::to_string(target_cone.center_x) + "," + std::to_string(target_cone.center_y) + " with prob: " + std::to_string(target_cone.prob));
		}
		else bDetected_cone = false;

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Target-Selections-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[Target-Selections-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Target-Selections-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[Target-Selections-VA] Unknown Exception!");
		return false;
	}
}
bool Pre_Land_chassis_position_VA(bool MountJob, bool OffloadJob, bbx target_cone, bbx target_hole, bool jobLog = false)
{
	try
	{
		if (target_cone.prob > 0)
		{
			if (MountJob)
			{
				if (AVG_CONE.get_count() < LDO_NCOUNT)
				{
					if (target_cone.label == 1 && target_cone.prob >= 70) AVG_CONE.update(target_cone);				
				}
				else
				{
					if (!AVG_CONE.get_set())
					{
						logMessage("Average Cone: (X: " + std::to_string(AVG_CONE.get_average().center_x) + " Y: " + std::to_string(AVG_CONE.get_average().center_y) + "),  prob: " + std::to_string(AVG_CONE.get_average().prob));
						if (jobLog) jobLogMessage("Average Cone: (X: " + std::to_string(AVG_CONE.get_average().center_x) + " Y: " + std::to_string(AVG_CONE.get_average().center_y) + "),  prob: " + std::to_string(AVG_CONE.get_average().prob));

						AVG_CONE.set();
					}
				}

			}
			else if (OffloadJob)
			{
				if (AVG_CONE.get_count() < CLPS_NCOUNT)
				{
					if (target_cone.prob >= 70) AVG_CONE.update(target_cone);					
				}
				else
				{
					if (!AVG_CONE.get_set())
					{
						logMessage("Average Cone: (X: " + std::to_string(AVG_CONE.get_average().center_x) + " Y: " + std::to_string(AVG_CONE.get_average().center_y) + "),  prob: " + std::to_string(AVG_CONE.get_average().prob));
						if (jobLog) jobLogMessage("Average Cone: (X: " + std::to_string(AVG_CONE.get_average().center_x) + " Y: " + std::to_string(AVG_CONE.get_average().center_y) + "),  prob: " + std::to_string(AVG_CONE.get_average().prob));
						AVG_CONE.set();
					}
				}

				//Take first "detectable" cones -- needs to be instant.
				if (OFFLOAD_CONE.get_count() < 5)
				{
					if (target_cone.prob >= 70 && target_cone.label == 1) OFFLOAD_CONE.update(target_cone);
				}
				else
				{
					if (!OFFLOAD_CONE.get_set())
					{
						logMessage("Offload Cone: (X: " + std::to_string(OFFLOAD_CONE.get_average().center_x) + " Y: " + std::to_string(OFFLOAD_CONE.get_average().center_y) + "),  prob: " + std::to_string(OFFLOAD_CONE.get_average().prob));
						if (jobLog) jobLogMessage("Offload Cone: (X: " + std::to_string(OFFLOAD_CONE.get_average().center_x) + " Y: " + std::to_string(OFFLOAD_CONE.get_average().center_y) + "),  prob: " + std::to_string(OFFLOAD_CONE.get_average().prob));
						OFFLOAD_CONE.set();
					}
				}
			}
		}

		if (target_hole.prob > 0)
		{
			if (OffloadJob)
			{
				if (AVG_HOLE.get_count() < CLPS_NCOUNT)
				{
					if (target_hole.prob > 70) AVG_HOLE.update(target_hole);					
				}
				else
				{
					if (!AVG_HOLE.get_set())
					{
						logMessage("Average Hole: (X: " + std::to_string(AVG_HOLE.get_average().center_x) + " Y: " + std::to_string(AVG_HOLE.get_average().center_y) + "),  prob: " + std::to_string(AVG_HOLE.get_average().prob));
						if (jobLog) jobLogMessage("Average Hole: (X: " + std::to_string(AVG_HOLE.get_average().center_x) + " Y: " + std::to_string(AVG_HOLE.get_average().center_y) + "),  prob: " + std::to_string(AVG_HOLE.get_average().prob));
						AVG_HOLE.set();
					}
				}
			}
		}

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Pre-Land-Chassis-Position-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Pre-Land-Chassis-Position-VA] Unknown Exception!");
		return false;
	}
}

bool Deviation_Output_VA(bool detected_xt, bool detected_cst, bool detected_unknown, bool MountJob, bbx target_hole, bbx target_cone, bbx target_guide, IDEAL_POS IP, int& tCntr_x, int& tCntr_y, int& tCntr_prob, int& tCone_x, int& tCone_y, int& tCone_prob, int& devOut_x, int& devOut_y, int& devOut_x_mm, int& devOut_y_mm, bool& usingLDO_Base, bool jobLog = false)
{
	try
	{
		//If target hole is present, deviation between target cone and hole.
		if (target_hole.prob > 0)
		{
			tCntr_x = target_hole.center_x;
			tCntr_y = target_hole.y + target_hole.h;
			tCntr_prob = target_hole.prob;
		}

		if (detected_xt)
		{
			if (target_cone.prob > 0)
			{
				tCone_x = target_cone.center_x;
				tCone_y = target_cone.center_y;
				tCone_prob = target_cone.prob;

				//if hole present,
				if (target_hole.prob > 0)
				{
					devOut_x = tCone_x - tCntr_x;
					devOut_y = tCntr_y - tCone_y;
				}
				//if hole absent,
				else
				{
					//2025.03.11 Modified.
					devOut_x = tCone_x - IP.CONE.center_x;
					devOut_y = tCone_y - IP.CONE.center_y;
				}
			}
		}
		else if (detected_cst)
		{
			if (target_guide.prob > 0)
			{
				tCone_x = target_guide.center_x;
				tCone_y = target_guide.center_y;
				tCone_prob = target_guide.prob;

				//if hole present,
				if (target_hole.prob > 0)
				{
					devOut_x = tCone_x - tCntr_x;
					devOut_y = tCntr_y - tCone_y;
				}
				//if hole absent,
				else
				{
					//2025.03.11 Modified.
					devOut_x = tCone_x - IP.GUIDE.center_x;
					devOut_y = tCone_y - IP.GUIDE.center_y;
				}
			}
		}
		else if (detected_unknown)
		{
			if (true)//(CHS_XT)
			{
				//use pre land average
				if (MountJob && AVG_CONE.get_count() >= LDO_NCOUNT)
				{
					target_cone = AVG_CONE.get_average();
					usingLDO_Base = true;
					if (target_cone.prob > 0)
					{
						tCone_x = target_cone.center_x;
						tCone_y = target_cone.center_y;
						tCone_prob = target_cone.prob;

						//if hole present,
						if (target_hole.prob > 0)
						{
							devOut_x = tCone_x - tCntr_x;
							devOut_y = tCntr_y - tCone_y;
						}
						//if hole absent,
						else
						{
							//2025.03.11 Modified.
							devOut_x = tCone_x - IP.CONE.center_x;
							devOut_y = tCone_y - IP.CONE.center_y;
						}
					}
				}
			}
		}
		//2025.03.11
		if (devOut_x != -10000 && devOut_y != -10000)
		{
			//Apply mm per pixel.
			//5mm per pixel.

			devOut_x_mm = devOut_x * 5;
			devOut_y_mm = devOut_y * 5;

		}
		logMessage("VA Dev Out: (X: " + std::to_string(devOut_x) + " Y: " + std::to_string(devOut_y) + ")");
		if (jobLog) jobLogMessage("VA Dev Out: (X: " + std::to_string(devOut_x) + " Y: " + std::to_string(devOut_y) + ")");
		logMessage("VA Dev Out in mm: (X: " + std::to_string(devOut_x_mm) + " Y: " + std::to_string(devOut_y_mm) + ")");
		if (jobLog) jobLogMessage("VA Dev Out in mm (X: " + std::to_string(devOut_x_mm) + " Y: " + std::to_string(devOut_y_mm) + ")");

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Deviation-Output-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Deviation-Output-VA] Unknown Exception!");
		return false;
	}
}

bool Debug_Landed_Trigger(bbx target_hole, bbx target_cone)
{
	try
	{
		bool debug_landed_trigger = false;
		if (target_cone.prob > 0)
		{
			if (target_hole.prob > 0)
			{
				if (target_hole.y + target_hole.h - target_cone.y > 0)
				{
					debug_landed_trigger = true;
				}
				else if (target_hole.y + target_hole.h < target_cone.y)
				{
					debug_landed_trigger = false;
				}
			}
			else if (target_cone.label == 2) //landed detected
			{
				debug_landed_trigger = true;
			}
			else if (target_cone.label == 1)
			{
				debug_landed_trigger = false;
			}
		}

		return debug_landed_trigger;
	}
	catch (std::exception& ex)
	{
		logMessage("[Debug-Landed-Trigger] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Debug-Landed-Trigger] Unknown Exception!");
		return false;
	}
}

bool LandOut_Detected_VA(bool MountJob, bool detected_xt, bool detected_cst, bbx target_hole, bbx target_cone, bool sprd_landed, bool detected_hole, int& devX, int& devY, bool usingLDO_Base, bool jobLog = false)
{
	try
	{
		if (MountJob)
		{
			//based on landed bit
			if (detected_xt)
			{
				if (sprd_landed)
				{
					//Case 1: if target cone is not landed.
					if (target_cone.label == 1)
					{
						if (LDO_Current_Count < 5) LDO_Current_Count++;
						else
						{
							if (!landout_detected)
							{
								logMessage("Landout detected with CONE object!");
								if (jobLog) jobLogMessage("Landout detected with CONE object!");
							}
							landout_detected = true;
						}
					}
					//Case 2: if target cone is landed
					else if (target_cone.label == 2 && !usingLDO_Base)
					{
						if (LandOK_Current_Count < 5) LandOK_Current_Count++;
						else
						{
							if (!landok_detected)
							{
								logMessage("LandOK detected with LANDED object!");
								if (jobLog) jobLogMessage("LandOK detected with LANDED object!");
							}
							landok_detected = true;
						}
					}
				}
				else
				{
					/*
					//SPRD Landed bit is ON, reset Landout and LandOK.
					LDO_Current_Count = 0;
					if (landout_detected)
					{
						logMessage("Landout Detection Released by SPRD LAND bit OFF");
						if (jobLog) jobLogMessage("Landout Detection Released by SPRD LAND bit OFF");
					}
					landout_detected = false;

					LandOK_Current_Count = 0;
					if (landok_detected)
					{
						logMessage("Landok Detection Released by SPRD LAND bit OFF");
						if (jobLog) jobLogMessage("Landok Detection Released by SPRD LAND bit OFF");
					}
					landok_detected = false;
					*/
				}
			}
			else if (detected_cst)
			{
				;
			}
			else //chassis not detected.
			{
				//No chassis detected.
				//only in live settings.
				//SPRD OFF but TWL Locked -- Remount.
				if (!SPRD_Landed && SPRD_TWL_Locked)
				{
					//SPRD Landed bit is ON, reset Landout and LandOK.
					LDO_Current_Count = 0;
					if (landout_detected)
					{
						logMessage("Landout Detection Released by SPRD LAND bit OFF");
						if (jobLog) jobLogMessage("Landout Detection Released by SPRD LAND bit OFF");
					}
					landout_detected = false;

					LandOK_Current_Count = 0;
					if (landok_detected)
					{
						logMessage("Landok Detection Released by SPRD LAND bit OFF");
						if (jobLog) jobLogMessage("Landok Detection Released by SPRD LAND bit OFF");
					}
					landok_detected = false;
				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[LandOut-Detected-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[LandOut-Detected-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[LandOut-Detected-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[LandOut-Detected-VA] Unknown Exception!");
		return false;
	}
}

bool CLPS_Detection_VA(bool OffloadJob, bbx target_hole, bbx target_cone, bool sprd_landed, bool jobLog = false)
{
	try
	{
		if (OffloadJob)
		{
			//2025.10.28 ; Disable CLPS detection based on cone movement due to Chassis-MoveOut before job finish.

			int diff_x = 0;
			int diff_y = 0;
			int LowerBound_diff_x = 0;
			int LowerBound_diff_y = 0;
			bool labelMatched = false;

			if (AVG_CONE.get_count() >= CLPS_NCOUNT)// && !clps_ok_detected)
			{
				if (target_cone.prob > 60)
				{
					auto CLPS_Base = AVG_CONE.get_average();

					diff_x = std::abs(CLPS_Base.center_x - target_cone.center_x);
					diff_y = std::abs(CLPS_Base.center_y - target_cone.center_y);
					UPPER_DIFF_X = diff_x;
					UPPER_DIFF_Y = diff_y;

					bool LowerCheckAvailable = (OFFLOAD_CONE.get_count() >= 5);
					auto CLPS_Lower_Base = bbx();
					LowerBound_diff_x = 0;
					LowerBound_diff_y = 0;	
					if (LowerCheckAvailable)
					{
						CLPS_Lower_Base = OFFLOAD_CONE.get_average();
						LowerBound_diff_x = std::abs(CLPS_Lower_Base.center_x - target_cone.center_x);
						//2026.01.08 -- check only going Lower than the base -- prevent detection via Chassis moveout.
						LowerBound_diff_y = target_cone.center_y - CLPS_Lower_Base.center_y;// std::abs(CLPS_Lower_Base.center_y - target_cone.center_y);
						if (CLPS_Lower_Base.label == target_cone.label) labelMatched = true;
					}

					LOWER_DIFF_X = LowerBound_diff_x;
					LOWER_DIFF_Y = LowerBound_diff_y;
					
					{
						if (AVG_CONE.get_set()) logMessage("Upper cone diff x,y: " + std::to_string(diff_x) + " , " + std::to_string(diff_y));
						if (OFFLOAD_CONE.get_set()) logMessage("Lower cone diff x,y: " + std::to_string(LowerBound_diff_x) + " , " + std::to_string(LowerBound_diff_y) + " , " + std::to_string(labelMatched));

						logMessage("Upper Thresholds: " + std::to_string(CLPS_NEAR_X_THRESHOLD) + " , " + std::to_string(CLPS_NEAR_Y_THRESHOLD));
						logMessage("Lower Thresholds: " + std::to_string(CLPS_LOWER_X_THRESHOLD) + " , " + std::to_string(CLPS_LOWER_Y_THRESHOLD));

						if (diff_y > CLPS_NEAR_Y_THRESHOLD && diff_x < CLPS_NEAR_X_THRESHOLD)
						{
							if (!clps_detected)
							{
								if (CLPS_Current_Count < CLPS_NCOUNT) CLPS_Current_Count++;

								if (CLPS_Current_Count >= CLPS_NCOUNT)
								{
									if (!clps_detected)
									{
										logMessage("CLPS Detected! by cone diff x,y: " + std::to_string(diff_x) + " , " + std::to_string(diff_y));
										if (jobLog) jobLogMessage("CLPS Detected! by cone diff x,y: " + std::to_string(diff_x) + " , " + std::to_string(diff_y));
									}
									clps_detected = true;

									//Reset ok status.
									if (clps_ok_detected)
									{
										CLPS_OK_Current_Count = 0;
										clps_ok_detected = false;
									}
								}
							}
						}
						//Lower Bound.
						else if (LowerBound_diff_x < CLPS_LOWER_X_THRESHOLD && LowerBound_diff_y > CLPS_LOWER_Y_THRESHOLD && labelMatched)
						{
							if (!clps_detected)
							{
								if (CLPS_Current_Count < CLPS_NCOUNT) CLPS_Current_Count++;

								if (CLPS_Current_Count >= CLPS_NCOUNT)
								{
									if (!clps_detected)
									{
										logMessage("CLPS Detected! by cone LOWER diff x,y: " + std::to_string(LowerBound_diff_x) + " , " + std::to_string(LowerBound_diff_y));
										if (jobLog) jobLogMessage("CLPS Detected! by cone LOWER diff x,y: " + std::to_string(LowerBound_diff_x) + " , " + std::to_string(LowerBound_diff_y));
									}
									clps_detected = true;

									//Reset ok status.
									if (clps_ok_detected)
									{
										CLPS_OK_Current_Count = 0;
										clps_ok_detected = false;
									}
								}
							}
						}

						else
						{
							if (clps_detected)
							{
								if (CLPS_Current_Count > 0) CLPS_Current_Count--;
								if (CLPS_Current_Count == 0)
								{
									clps_detected = false;
								}
							}
						}
					}
				}
				else
				{
					//if inference can be trusted, then non detection can be an indicator of CLPS.
				}
			}

			//2025.03.11
			//CLPS-OK Logic.
			//clps_ok_detected = false;
			if (true)//!sprd_landed && !clps_detected)
			{
				//if valid cornercastings and cone (label==1) detected, then clps-ok.
				if (target_hole.prob > 70 && (target_cone.prob > 70 && target_cone.label == 1))
				{
					auto diff_hc_y = target_cone.center_y - target_hole.center_y;
					if (diff_hc_y > 20)
					{
						if ((diff_y > CLPS_NEAR_Y_THRESHOLD && diff_x < CLPS_NEAR_X_THRESHOLD) || (LowerBound_diff_x < CLPS_LOWER_X_THRESHOLD && LowerBound_diff_y > CLPS_LOWER_Y_THRESHOLD && labelMatched))
						{
							;
						}
						else
						{
							if (CLPS_OK_Current_Count < 3) CLPS_OK_Current_Count++;
							if (CLPS_OK_Current_Count >= 3)
							{
								if (!clps_ok_detected)
								{
									logMessage("VA CLPS-OK Detected! by Cone and Hole, diff_y: " + std::to_string(diff_y));
									if (jobLog) jobLogMessage("VA CLPS-OK Detected! by Cone and Hole, diff_y: " + std::to_string(diff_y));
								}
								clps_ok_detected = true;

								if (clps_detected)
								{
									CLPS_Current_Count = 0;
									clps_detected = false;
								}
							}
						}
					}
				}
				//2025.08.23
				else if (target_cone.prob > 70 && target_cone.label == 1)
				{
					if ((diff_y > CLPS_NEAR_Y_THRESHOLD && diff_x < CLPS_NEAR_X_THRESHOLD) || (LowerBound_diff_x < CLPS_LOWER_X_THRESHOLD && LowerBound_diff_y > CLPS_LOWER_Y_THRESHOLD && labelMatched))
					{
						;
					}
					else
					{
						if (CLPS_OK_Current_Count < 3) CLPS_OK_Current_Count++;
						if (CLPS_OK_Current_Count >= 3)
						{
							if (!clps_ok_detected)
							{
								logMessage("VA CLPS-OK Detected! by Cone, no Hole, prob: " + std::to_string(target_cone.prob));
								if (jobLog) jobLogMessage("VA CLPS-OK Detected! by Cone, no Hole, prob: " + std::to_string(target_cone.prob));
							}
							clps_ok_detected = true;

							if (clps_detected)
							{
								CLPS_Current_Count = 0;
								clps_detected = false;
							}
						}
					}
				}
				else if (clps_ok_detected)
				{
					if (target_cone.prob > 70 && target_cone.label == 2)
					{
						if (clps_ok_detected)
						{
							logMessage("VA CLPS-OK Released! by Landed");
							if (jobLog) jobLogMessage("VA CLPS-OK Released! by Landed");
						}
						clps_ok_detected = false;
						CLPS_OK_Current_Count = 0;
					}	
				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[CLPS-Detection-VA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[CLPS-Detection-VA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[CLPS-Detection-VA] Unknown Exception!");
		if (jobLog) jobLogMessage("[CLPS-Detection-VA] Unknown Exception!");
		return false;
	}
}


//Chassis End Extraction in case of VA non detection.
pcl::PointXYZ ChassisEndPoint_Extract(pcl::PointCloud<pcl::PointXYZ>::Ptr input, bool isLeftSensor, std::string savePath = std::string(""))
{
	try
	{
		// ---- Tunables (relaxed for robustness) ----
		float ROI_X_Min = -2000.f, ROI_X_Max = 2000.f;
		float ROI_Y_Min = -1500.f, ROI_Y_Max = -500.f;
		float ROI_Z_Min = 1500.f, ROI_Z_Max = 5000.f;

		if (isLeftSensor) { ROI_X_Max = -800.f; }
		else { ROI_X_Min = 800.f; }

		float VOX_MM = 5.0f;
		int   SOR_K = 30;
		float SOR_STD = 1.0f;        // previously 1.5

		int pcThreshold = 500;

		pcl::PointXYZ ChassisPos(-10000, -10000, -10000);

		pcl::PointCloud<pcl::PointXYZ>::Ptr pass_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pc_passThrough(input, ROI_X_Min, ROI_X_Max, ROI_Y_Min, ROI_Y_Max, ROI_Z_Min, ROI_Z_Max, pass_filtered);

		if (savePath != std::string(""))
			pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_PT.pcd", *pass_filtered);

		//2. Voxel DownSample
		pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pc_VoxelDown(pass_filtered, VOX_MM, VOX_MM, VOX_MM, voxel_filtered);

		if (savePath != std::string(""))
			pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_VX.pcd", *voxel_filtered);

		//3. Statistical Outlier Removal
		pcl::PointCloud<pcl::PointXYZ>::Ptr sor_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pc_NoiseFilter(voxel_filtered, SOR_K, SOR_STD, sor_filtered);

		if (savePath != std::string(""))
			pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_NF.pcd", *sor_filtered);

		//4. Slice from minZ to maxZ in 50mm slices to get valid tailEnd of the chassis.
		int minZ, maxZ;
		minZ = get_valid_min_z(sor_filtered);
		//get_max_min_z(sor_filtered, ref(maxZ), ref(minZ));
		int slice_step = 50;
		//minZ to 300mm (10 slices?)
		for (int i = 0; i < 20; i++)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice(new pcl::PointCloud<pcl::PointXYZ>);
			pc_passThrough(sor_filtered, minZ + i * slice_step, minZ + (i + 1) * slice_step, "z", slice);

			if (slice->points.size() > pcThreshold)
			{
				if (savePath != std::string(""))
					pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_slct_slice_z_i_" + std::to_string(i) + "pc_" + std::to_string(slice->points.size()) + ".pcd", *slice);

				int refMinZ, refMaxZ;
				int refMinX, refMaxX;
				int refMinY, refMaxY;
				get_max_min_z(slice, refMaxZ, refMinZ);
				get_center_of_cloud(*slice, ref(refMaxX), ref(refMinX), ref(refMaxY), ref(refMinY), ref(refMaxZ), ref(refMinZ));

				if (isLeftSensor)
				{
					ChassisPos = pcl::PointXYZ(refMaxX, (refMaxY + refMinY) / 2, refMinZ);
				}
				else
				{
					ChassisPos = pcl::PointXYZ(refMinX, (refMaxY + refMinY) / 2, refMinZ);
				}
			
				//ChassisPosition = refMinZ;
				//refChassisPosition = pcl::PointXYZ((refMaxX + refMinX) / 2, (refMaxY + refMinY) / 2, refMinZ);

				if (savePath != std::string(""))
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr cpos(new pcl::PointCloud<pcl::PointXYZ>);
					cpos->points.push_back(ChassisPos);

					pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_Pos.pcd", *cpos);
				}
				break;
			}
			else
			{
				if (savePath != std::string(""))
					pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_slice_z_i_" + std::to_string(i) + "pc_" + std::to_string(slice->points.size()) + ".pcd", *slice);

			}
		}

		return ChassisPos;
	}
	catch (std::exception& ex)
	{
		logMessage("[Chassis-End-Point-Extract] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[Chassis-End-Point-Extract] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}		
}
double calculate_average(const std::vector<int>& vec, int index) {
	// Check if the vector is empty to prevent division by zero
	if (vec.empty()) {
		return 0.0;
	}

	// Calculate the sum using std::accumulate
	// The third argument (0.0) sets the initial value and result type to double
	double sum = std::accumulate(vec.begin(), vec.begin() + (index - 1), 0.0);

	// Calculate the average
	double average = sum / index;// vec.size();

	return average;
}
pcl::PointXYZ Chassis_Extract(pcl::PointCloud<pcl::PointXYZ>::Ptr input, bool isLeftSensor, int ref_hole_height, std::string savePath = std::string(""))
{
	try
	{
		float ROI_Z = 2700.f;
		float Z_Range = 600.f;
		float ROI_Y = -850.f;
		float Y_UpperRange = 600.f; //at least 1m above from expected height.
		float Y_LowerRange = 300.f;

		float ROI_X = -1100.f;
		if (!isLeftSensor) ROI_X = 1100.f;
		float X_Range = 400.f;

		float VOX_MM = 5.0f;
		int   SOR_K = 20;
		float SOR_STD = 1.0f;        // previously 1.5

		int pcThreshold = 500;

		pcl::PointXYZ ChassisPos(-10000, -10000, -10000);

		pcl::PointCloud<pcl::PointXYZ>::Ptr pass_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		if (isLeftSensor) pc_passThrough(input, ROI_X - X_Range, ROI_X + X_Range/2, ROI_Y - Y_LowerRange, ROI_Y + Y_UpperRange, ROI_Z - Z_Range, ROI_Z + Z_Range, pass_filtered);
		else pc_passThrough(input, ROI_X - X_Range / 2, ROI_X + X_Range, ROI_Y - Y_LowerRange, ROI_Y + Y_UpperRange, ROI_Z - Z_Range, ROI_Z + Z_Range, pass_filtered);

		if (savePath != std::string(""))
			pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_PT.pcd", *pass_filtered);

		int local_max_z, local_min_z;
		local_min_z = get_valid_min_z(pass_filtered);
		if (abs(local_min_z - ROI_Z) > 200)
		{
			ROI_Z = round(local_min_z);

			if (isLeftSensor) pc_passThrough(input, ROI_X - X_Range, ROI_X + X_Range / 2, ROI_Y - Y_LowerRange, ROI_Y + Y_UpperRange, ROI_Z - Z_Range, ROI_Z + Z_Range, pass_filtered);
			else pc_passThrough(input, ROI_X - X_Range / 2, ROI_X + X_Range, ROI_Y - Y_LowerRange, ROI_Y + Y_UpperRange, ROI_Z - Z_Range, ROI_Z + Z_Range, pass_filtered);

			if (savePath != std::string(""))
				pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_PT_Adjusted.pcd", *pass_filtered);
		}

		//2. Voxel DownSample
		pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pc_VoxelDown(pass_filtered, VOX_MM, VOX_MM, VOX_MM, voxel_filtered);

		if (savePath != std::string(""))
			pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_VX.pcd", *voxel_filtered);

		//3. Statistical Outlier Removal
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr sor_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pc_NoiseFilter(voxel_filtered, SOR_K, SOR_STD, sor_filtered);

		if (savePath != std::string(""))
			pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_NF.pcd", *sor_filtered);
		

		//4. Slice through y to check landed status // mini clustering here.
		int slice_width = 30;
		std::vector<int> slice_z_widths = {};
		std::vector<int> slice_x_widths = {};
		std::vector<int> slice_points = {};
		std::vector<int> slice_cy = {};

		//States.
		bool coneOnly = false;
		bool separated = false;
		bool landed = false;

		//Cone-Only Check
		int top_empty_counter = 0;
		//if top empty counter hits 5, then cone-only.
		//top empty counter only counts until first valid slice.

		//Separated_Check
		bool gap_start_enable = false;
		int gap_start_i = -1;
		int gap_end_i = -1;


		int i = 0;

		//logMessage("MinY: " + std::to_string(ROI_Y - Y_LowerRange) + " MaxY: " + std::to_string(ROI_Y + Y_UpperRange));
		for (int y = ROI_Y + Y_UpperRange; y > ROI_Y - Y_LowerRange; y -= slice_width)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice(new pcl::PointCloud<pcl::PointXYZ>);
			pc_passThrough(sor_filtered, y - slice_width, y, "y", slice);
			//logMessage("slice: " + std::to_string(slice->points.size()));
			//logMessage("MinY: " + std::to_string(y - slice_width) + " MaxY: " + std::to_string(y));
			if (slice->points.size() > 50)
			{
				//Cone-Only Check
				if (i < 6) top_empty_counter = 0;

				//Gap Check
				if (!gap_start_enable) gap_start_enable = true;
				if (gap_start_i != -1 && i - gap_start_i < 4) gap_start_i = -1; //reset gap if small gap.
				else if (gap_start_i != -1 && gap_end_i == -1) gap_end_i = i;

				int maxZ, minZ;
				get_max_min_z(slice, maxZ, minZ);
				int width = maxZ - minZ;
				slice_z_widths.push_back(width);

				int maxX, minX;
				get_max_min_x(slice, maxX, minX);
				int x_width = maxX - minX;
				slice_x_widths.push_back(x_width);

				slice_points.push_back(slice->points.size());
				slice_cy.push_back(y - slice_width / 2);
				//if (savePath != std::string(""))
				//	pcl::io::savePCDFileBinaryCompressed(savePath + "_CEnd_y_i_" + std::to_string(i) + "_zW_" + std::to_string(width) + ".pcd", *slice);
			}
			else
			{
				if (i < 6) top_empty_counter++;


				if (gap_start_i == -1 && gap_start_enable) gap_start_i = i;
			}
			i++;
		}
		
		if (top_empty_counter >= 5)
		{
			coneOnly = true;
			logMessage("Chassis-End: Cone-Only detected.");
		}

		int y_gap_lower_level = -10000;
		if (gap_start_i != -1 && gap_end_i != -1)
		{
			separated = true;
			y_gap_lower_level = slice_cy.at(gap_end_i) - 200;

			logMessage("Chassis-End: Separated detected. Gap from slice " + std::to_string(gap_start_i) + " to " + std::to_string(gap_end_i) + ", lower level at Y: " + std::to_string(y_gap_lower_level));
		}

		if (!coneOnly && !separated)
		{
			landed = true;
			logMessage("Chassis-End: Landed detected.");
		}

		//Y Level Sets.

		int y_sep_level = -10000;
		bool valid_y_sep = false;

		if (coneOnly)
		{
			//From first valid data.
			y_sep_level = slice_cy.at(0);// -200; //accounting for cone.
			valid_y_sep = true;
		}
		else if (separated)
		{
			//Gap from lower level.
			y_sep_level = y_gap_lower_level;
			valid_y_sep = true;
		}
		else
		{
			int avg_z_width = calculate_average(slice_z_widths, 5);
			int avg_x_width = calculate_average(slice_x_widths, 5);
			logMessage("Chassis-End Avg X,Z Width: " + std::to_string(avg_x_width) + " , " + std::to_string(avg_z_width));

			if (avg_z_width > 400)
			{
				//Use z diff.
				for (int j = 0; j < slice_z_widths.size(); j++)
				{
					if (slice_z_widths.at(j) < avg_z_width)
					{
						y_sep_level = slice_cy.at(j) - 100;
						//valid_y_sep = true;
						break;
					}
				}
			}
			else
			{
				for (int j = 0; j < slice_x_widths.size(); j++)
				{
					if (slice_x_widths.at(j) < avg_x_width)
					{
						y_sep_level = slice_cy.at(j) - 100;
						//valid_y_sep = true;
						break;
					}
				}
			}

			if (y_sep_level < -950)
			{
				y_sep_level = -950;
			}	
		}

		//now bypass any levels. if ref_hole_height is given.
		if (ref_hole_height != -10000)
		{
			y_sep_level = ref_hole_height - 50;
			valid_y_sep = true;
		}

		//Add this to vec
		if (valid_y_sep) chassis_sep_height.push_back(y_sep_level);
		else
		{
			//use the latest one.
			y_sep_level = chassis_sep_height.back() - 100;
		}

		//5. Slice from minZ to maxZ in 50mm slices to get valid tailEnd of the chassis.
		int minZ, maxZ;
		minZ = get_valid_min_z(sor_filtered);
		//get_max_min_z(sor_filtered, ref(maxZ), ref(minZ));
		int slice_step = 50;
		//minZ to 300mm (10 slices?)
		for (int i = 0; i < 20; i++)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice(new pcl::PointCloud<pcl::PointXYZ>);
			pc_passThrough(sor_filtered, minZ + i * slice_step, minZ + (i + 1) * slice_step, "z", slice);

			if (slice->points.size() > pcThreshold)
			{
				if (savePath != std::string(""))
					pcl::io::savePCDFileBinaryCompressed(savePath + "_CE_slct_slz_i_" + std::to_string(i) + "pc_" + std::to_string(slice->points.size()) + ".pcd", *slice);

				int refMinZ, refMaxZ;
				int refMinX, refMaxX;
				int refMinY, refMaxY;
				get_max_min_z(slice, refMaxZ, refMinZ);
				get_center_of_cloud(*slice, ref(refMaxX), ref(refMinX), ref(refMaxY), ref(refMinY), ref(refMaxZ), ref(refMinZ));

				if (isLeftSensor)
				{
					ChassisPos = pcl::PointXYZ(refMaxX, (refMaxY + refMinY) / 2, refMinZ);
				}
				else
				{
					ChassisPos = pcl::PointXYZ(refMinX, (refMaxY + refMinY) / 2, refMinZ);
				}

				if (landed) ChassisPos.y = y_sep_level;

				//ChassisPosition = refMinZ;
				//refChassisPosition = pcl::PointXYZ((refMaxX + refMinX) / 2, (refMaxY + refMinY) / 2, refMinZ);

				if (savePath != std::string(""))
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr cpos(new pcl::PointCloud<pcl::PointXYZ>);
					cpos->points.push_back(ChassisPos);

					if(coneOnly)
						pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_Pos_ConeOnly.pcd", *cpos);
					else if(separated)
						pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_Pos_Separated.pcd", *cpos);
					else if(landed)
						pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_Pos_Landed.pcd", *cpos);	
				}
				break;
			}
			else
			{
				if (savePath != std::string(""))
					pcl::io::savePCDFileBinaryCompressed(savePath + "_ChassisEnd_slice_z_i_" + std::to_string(i) + "pc_" + std::to_string(slice->points.size()) + ".pcd", *slice);

			}
		}

		return ChassisPos;
	}
	catch (std::exception& ex)
	{
		logMessage("[Chassis-End-Point-Extract] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[Chassis-End-Point-Extract] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
}

bool Target_PointCloud_Extraction(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bool isLeftSensor, bbx target_hole, bbx target_cone, bbx target_guide, bool sprd_landed, IDEAL_POS IP, int x_limit, int z_limit, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcHole, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcCone, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcGuide, bool avgUsable = false, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (target_hole.prob > 0)
		{
			std::string bPath = (save_path != std::string("")) ? save_path + "_Hole" : "";
			//logMessage(std::to_string(target_hole.x) + " " + std::to_string(target_hole.y) + " , " + std::to_string(x_limit) + " , " + std::to_string(z_limit));
			pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud <pcl::PointXYZ>);
			temp = map_box_to_pc_noisefiltertest(pointCloud, target_hole, x_limit, z_limit, bPath);
			pcHole = init_filter(temp, isLeftSensor, 100, 200, 200, 100, "Hole", save_path);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcHole.pcd", *pcHole);
		}
		if (target_cone.prob > 0)
		{
			if (target_cone.label == 1)
			{
				std::string bPath = (save_path != std::string("")) ? save_path + "_Cone" : "";
				pcCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, x_limit, z_limit, bPath);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcCone.pcd", *pcCone);
			}
			else if (target_cone.label == 2) //Discard "Guide" detection on T-Mini.
			{
				//modify detected landed area.
				if (target_hole.prob > 0)
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcPreLanded.pcd", *pcPreCone);

					bbx modified_landed = target_cone;

					if (target_hole.y + target_hole.h > target_cone.y)
					{
						auto delta_y = target_hole.y + target_hole.h - target_cone.y;
						modified_landed.y = target_hole.y + target_hole.h;
						modified_landed.h -= delta_y;
					}
					else
					{
						//preset y down?
						int preset_y_hole = 20;
						modified_landed.y = target_cone.y + preset_y_hole;
						modified_landed.h = target_cone.h - preset_y_hole;
					}
					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcLanded.pcd", *pcCone);
				}
				else
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, target_cone, x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcPreLanded.pcd", *pcPreCone);

					//preset y down?
					int preset_y_hole = 20;
					bbx modified_landed = target_cone;
					modified_landed.y = target_cone.y + preset_y_hole;
					modified_landed.h = target_cone.h - preset_y_hole;

					std::string bPath2 = (save_path != std::string("")) ? save_path + "_preOff_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_preOff_pcLanded.pcd", *pcCone);
				}
			}
			else
			{
				; //Guide Type.
			}
		}
		else
		{
			if (AVG_CONE.get_average().prob > 0 && avgUsable)
			{
				if (AVG_CONE.get_average().label == 1)
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_Avg_Cone" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, AVG_CONE.get_average(), x_limit, z_limit, bPath);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_Avg_pcCone.pcd", *pcCone);
				}
				else
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_Avg_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, AVG_CONE.get_average(), x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pc_Avg_PreLanded.pcd", *pcPreCone);

					bbx modified_landed = AVG_CONE.get_average();
					if (target_hole.prob > 0)
					{
						if (target_hole.y + target_hole.h > AVG_CONE.get_average().y)
						{
							auto delta_y = target_hole.y + target_hole.h - AVG_CONE.get_average().y;
							modified_landed.y = target_hole.y + target_hole.h;
							modified_landed.h -= delta_y;
						}
						else
						{
							//preset y down?
							int preset_y_hole = 20;
							modified_landed.y = AVG_CONE.get_average().y + preset_y_hole;
							modified_landed.h = AVG_CONE.get_average().h - preset_y_hole;
						}
					}
					else
					{
						//preset y down?
						int preset_y_hole = 20;
						modified_landed.y = AVG_CONE.get_average().y + preset_y_hole;
						modified_landed.h = AVG_CONE.get_average().h - preset_y_hole;
					}
					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Avg_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_Avg_pcLanded.pcd", *pcCone);
				}
			}
			else
			{
				
				int ref_hole_height = -10000;
				if (pcHole->points.size() > 0)
				{
					ref_hole_height = get_valid_min_y(pcHole);
				}

				auto t_start = std::chrono::high_resolution_clock::now();
				auto refPoint = Chassis_Extract(pointCloud, isLeftSensor, ref_hole_height, std::string(""));// save_path);
				logMessage("Chassis Extracted time: " + std::to_string(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count()) + " ms");

				//X: 300mm, Z: 300mm, Y: - 200mm. //only consider height below the sample.
				if (refPoint.x != -10000)
				{
					if (isLeftSensor)
					{
						pc_passThrough(pointCloud,
							refPoint.x - 300, refPoint.x + 200,
							refPoint.y - 300, refPoint.y,
							refPoint.z - 100, refPoint.z + 300, pcCone);

						if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_EXT_pcCone.pcd", *pcCone);
					}
					else
					{
						pc_passThrough(pointCloud,
							refPoint.x - 200, refPoint.x + 300,
							refPoint.y - 300, refPoint.y,
							refPoint.z - 100, refPoint.z + 300, pcCone);
						if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_EXT_pcCone.pcd", *pcCone);
					}
				}
				else
				{
					logMessage("Invalid ChassisPoint Output. Can't extract target cone pointcloud.");
				}
			}

			//extract based on preset.
			//preset
			/*
			if (!sprd_landed)
			{
				bbx preset_cone = IP.CONE;
				std::string bPath = (save_path != std::string("")) ? save_path + "_PreSet_Cone" : "";
				pcCone = map_box_to_pc_noisefiltertest(pointCloud, preset_cone, x_limit, z_limit, bPath);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcPresetCone.pcd", *pcCone);
			}
			else //assume landed.
			{
				bbx preset_cone = IP.LANDED;
				//modify detected landed area.
				if (target_hole.prob > 0)
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreSetLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, preset_cone, x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcPreSetLanded.pcd", *pcCone);

					bbx modified_landed = preset_cone;
					auto delta_y = target_hole.y + target_hole.h - preset_cone.y;
					modified_landed.y = target_hole.y + target_hole.h;
					modified_landed.h -= delta_y;

					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcLanded.pcd", *pcCone);
				}
				else
				{
					std::string bPath = (save_path != std::string("")) ? save_path + "_PreLanded" : "";
					auto pcPreCone = map_box_to_pc_noisefiltertest(pointCloud, preset_cone, x_limit, z_limit, bPath);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcPreLanded.pcd", *pcCone);

					//preset y down?
					int preset_y_hole = 20;
					bbx modified_landed = preset_cone;
					modified_landed.y = preset_cone.y + preset_y_hole;
					modified_landed.h = preset_cone.h - preset_y_hole;
					std::string bPath2 = (save_path != std::string("")) ? save_path + "_Landed" : "";
					pcCone = map_box_to_pc_noisefiltertest(pointCloud, modified_landed, x_limit, z_limit, bPath2);
					if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcLanded_preset.pcd", *pcCone);
				}
			}
			*/
		}

		if (target_guide.prob > 0)
		{
			pcGuide = map_box_to_pc(pointCloud, target_guide);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcGuide.pcd", *pcGuide);
		}

		if (jobLog) jobLogMessage("pcHole count: " + std::to_string(pcHole->points.size()));
		if (jobLog) jobLogMessage("pcCone count: " + std::to_string(pcCone->points.size()));
		if (jobLog) jobLogMessage("pcGuide count: " + std::to_string(pcGuide->points.size()));

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Target-PointCloud-Extraction] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Target-PointCloud-Extraction] Unknown Exception!");
		return false;
	}
}

//2025.08.06
//CLPS Detection using only 3D Point Cloud data.
//input: 3d base point cloud
//output: Separated or combined.
bool CLPS_Detection_PCA_Only(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, bool isLeftSensor, int& devY, std::string savePath = std::string(""))
{
	auto t_start = std::chrono::high_resolution_clock::now();
	bool blnSeparated = false;
	try
	{
		if (TZ_Offload_Cycle)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcPassThrough(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcVoxel(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcNoiseFilter(new pcl::PointCloud<pcl::PointXYZ>);

			//step1: Extract ROI based on preset
			int x_min_limit = 450;
			int x_max_limit = 450 + 200 + 1200;
			if (isLeftSensor)
			{
				x_min_limit = -1 * x_max_limit; //invert x limit for right sensor.
				x_max_limit = -450;
			}
			int y_min_limit = -1200;
			int y_max_limit = 500;

			int z_min_limit = 1500;
			int z_max_limit = 4000;

			pc_passThrough(pointcloud, x_min_limit, x_max_limit, y_min_limit, y_max_limit, z_min_limit, z_max_limit, pcPassThrough);
			if (savePath != std::string("") && pcPassThrough->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(savePath + "_pcPassThrough.pcd", *pcPassThrough);

			//step2: Noise filter
			pc_NoiseFilter(pcPassThrough, 30, 0.7, pcNoiseFilter);
			if (savePath != std::string("") && pcNoiseFilter->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(savePath + "_pcNoiseFilter.pcd", *pcNoiseFilter);

			//step3: Voxel Down Filter
			pc_VoxelDown(pcNoiseFilter, 50.0f, 50.0f, 50.0f, pcVoxel);
			if (savePath != std::string("") && pcVoxel->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(savePath + "_pcVoxel.pcd", *pcVoxel);

			//step4: clustering
#pragma region Clustering
//auto clustering_time = std::chrono::high_resolution_clock::now();

			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
			tree->setInputCloud(pcVoxel);

			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
			ec.setInputCloud(pcVoxel);
			//ec.setClusterTolerance(12);
			ec.setClusterTolerance(100); //100mm Euclidean distance (between two points) tolerance for clustering.
			//ec.setMinClusterSize(1000);
			ec.setMinClusterSize(200);
			ec.setMaxClusterSize(20000);
			ec.setSearchMethod(tree);
			ec.extract(cluster_indices);

			int i = 0;
			int extra_range = 10;
			std::vector<pcl::PointCloud<pcl::PointXYZ>> extracted_clusters;
			std::vector<cloud_data> extracted_clusters_data;
			for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
				for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
				{
					cloud_cluster->points.push_back(pcVoxel->points[*pit]);
				}
				//extracted_clusters.push_back(*cloud_cluster);

				//auto cloud_extract_time = std::chrono::high_resolution_clock::now();
				int c_max_x = 0, c_min_x = 0, c_max_y = 0, c_min_y = 0, c_max_z = 0, c_min_z = 0;
				auto center = get_center_of_cloud(*cloud_cluster, c_max_x, c_min_x, c_max_y, c_min_y, c_max_z, c_min_z);

				//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_extrated(new pcl::PointCloud<pcl::PointXYZ>);
				//pc_passThrough(pcPassThrough, c_min_x - extra_range, c_max_x + extra_range, c_min_y, c_max_y, c_min_z - extra_range, c_max_z + extra_range, cloud_extrated);

				cloud_data cloud_data_extracted;
				cloud_data_extracted.pc_ptr = cloud_cluster;
				cloud_data_extracted.max_x = c_max_x;
				cloud_data_extracted.min_x = c_min_x;
				cloud_data_extracted.max_y = c_max_y;
				cloud_data_extracted.min_y = c_min_y;
				cloud_data_extracted.max_z = c_max_z;
				cloud_data_extracted.min_z = c_min_z;

				extracted_clusters_data.push_back(cloud_data_extracted);
				extracted_clusters.push_back(*cloud_cluster);

				//auto cloud_extract_Endtime = std::chrono::high_resolution_clock::now();
				//auto cloud_extract_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cloud_extract_Endtime - cloud_extract_time);
				//leave_a_message(log_heading + "Cloud extract algorithm= " + std::to_string(cloud_extract_duration.count()) + "msec", std::chrono::system_clock::now());

				if (savePath != std::string(""))pcl::io::savePCDFileBinaryCompressed(savePath + "_cluster_" + std::to_string(i) + ".pcd", *cloud_cluster);
				i++;
			}

			//auto clustering_Endtime = std::chrono::high_resolution_clock::now();
			//auto clustering_duration = std::chrono::duration_cast<std::chrono::milliseconds>(clustering_Endtime - clustering_time);
			//logMessage("Clustering algorithm= " + std::to_string(clustering_duration.count()) + "msec");
#pragma endregion

//Step5: Get Min H, Max H from biggest two clusters and check dev Y.
			if (extracted_clusters_data.size() > 0)
			{
				if (extracted_clusters_data.size() == 1)
				{
					//empty chassis or loaded chassis.
					if (extracted_clusters_data.at(0).max_y > 0)
					{
						//cluster data above sensor height -> loaded chassis.
					}
					else
					{
						//cluster data below sensor height -> empty chassis.
					}
				}
				else if (extracted_clusters_data.size() >= 2)
				{
					//take biggest two clusters.
					std::vector<cloud_data> two_biggest_clusters{};
					two_biggest_clusters.push_back(extracted_clusters_data.at(0));
					two_biggest_clusters.push_back(extracted_clusters_data.at(1));

					std::sort(two_biggest_clusters.begin(), two_biggest_clusters.end(), [](const cloud_data& a, const cloud_data& b) {
						return a.max_y > b.max_y; // Sort by max_y (height)
						});

					auto container_cluster = two_biggest_clusters.at(0);
					auto chassis_cluster = two_biggest_clusters.at(1);

					if (savePath != std::string(""))pcl::io::savePCDFileBinaryCompressed(savePath + "_container_clust.pcd", *container_cluster.pc_ptr);
					if (savePath != std::string(""))pcl::io::savePCDFileBinaryCompressed(savePath + "_chassis_clust.pcd", *chassis_cluster.pc_ptr);

					devY = static_cast<int>(container_cluster.min_y - chassis_cluster.max_y);
					if (devY > 200)
					{
						logMessage("CLPS-OK-PCA-ONLY: Chassis separated");
						if (clps_ok_detected_pca_only_count >= CLPS_NCOUNT)
						{
							if (!clps_ok_detected_pca_only)
							{
								logMessage("CLPS-OK Detected by PCA-Only");
								clps_ok_detected_pca_only = true;
							}
						}
						else
						{
							if (clps_ok_detected_pca_only_count < 20) clps_ok_detected_pca_only_count++;
						}
						blnSeparated = true;
					}
					else
					{
						if (clps_ok_detected_pca_only_count == 0)
						{
							if (clps_ok_detected_pca_only)
							{
								;
								//logMessage("CLPS-OK Detection Released by PCA-Only");
								//clps_ok_detected_pca_only = false;
							}
						}
						else
						{
							//if (clps_ok_detected_pca_only_count > 0) clps_ok_detected_pca_only_count--;
						}

						blnSeparated = false;
					}
				}
			}

			auto t_stop = std::chrono::high_resolution_clock::now();
			auto t_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t_stop - t_start);
			logMessage("PCA-ONLY CLPS Proc Time= " + std::to_string(t_dur.count()) + "msec");
		}
		return blnSeparated;
	}
	catch (std::exception& ex)
	{
		logMessage("[CLPS-Detection-PCA-Only] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[CLPS-Detection-PCA-Only] Unknown Exception!");
		return false;
	}
}

bool CLPS_Detection_PCA(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, bool isLeftSensor, bbx target_hole, pcl::PointXYZ hole_pos, int g_data_limit, int t_data_limit, bool detected_hole, bool sprd_landed, int& pcUnderHole, int& clps_y_level, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (TZ_Offload_Cycle)
		{
			//Hole detected, but not chassis.
			if (detected_hole)
			{
				//if hole lifted above certain level.
				clps_y_level = 414 - (414 / 3);
				if (AVG_HOLE.get_count() >= CLPS_NCOUNT)
				{
					auto bb = AVG_HOLE.get_average();
					clps_y_level = bb.y - bb.h * 2.5;
				}
				if (jobLog) jobLogMessage("PCA-CLPS Y Level Threshold: " + std::to_string(clps_y_level));
				if (target_hole.y < clps_y_level)
				{
					//extract region beneath the detected hole.
					bbx targetRegion;
					
					targetRegion.x = target_hole.x;
					targetRegion.y = target_hole.y + target_hole.h;
					targetRegion.w = target_hole.w;
					targetRegion.h = target_hole.h * 2;
					
					if (isLeftSensor)
					{
						targetRegion.w *= 2;
					}
					else
					{
						targetRegion.x -= targetRegion.w;
						targetRegion.w *= 2;
					}
					
					targetRegion.center_x = targetRegion.x + targetRegion.w / 2;
					targetRegion.center_y = targetRegion.y + targetRegion.h / 2;

					//then extract.
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcHoleBeneath(new pcl::PointCloud<pcl::PointXYZ>);
					pcHoleBeneath = map_box_to_pc(pointCloud, targetRegion);
					//if (save_path != std::string(""))pcl::io::savePCDFileBinaryCompressed(save_path + "_pcHoleBeneath.pcd", *pcHoleBeneath);

					//Do a pre filter 
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcFilterZ(new pcl::PointCloud <pcl::PointXYZ>);
					pc_passThrough(pcHoleBeneath, 0, g_data_limit, "z", pcFilterZ);
					pcl::PointCloud<pcl::PointXYZ>::Ptr pcTargetBase(new pcl::PointCloud <pcl::PointXYZ>);
					pc_passThrough(pcFilterZ, 0, t_data_limit, "x", pcTargetBase);

					//Added filter
					if (hole_pos.x > -10000 && hole_pos.x != 0)
					{
						pc_passThrough(pcTargetBase, hole_pos.z - 100, hole_pos.z + 200, "z", pcTargetBase);
						pc_passThrough(pcTargetBase, hole_pos.x - 200, hole_pos.x + 200, "x", pcTargetBase);
					}

					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_pcHoleBeneath_Target_" + std::to_string(pcTargetBase->points.size()) + ".pcd", *pcTargetBase);

					pcUnderHole = pcTargetBase->points.size();

					if (pcTargetBase->points.size() > 1000)
					{
						if (clps_current_count_pca < CLPS_NCOUNT) clps_current_count_pca++;
						else
						{
							if (!clps_detected_pca)
							{
								logMessage("CLPS Detected by PCA");
								if (jobLog) jobLogMessage("CLPS Detected by PCA");
								clps_detected_pca = true;
							}
						}

						if (clps_ok_current_count_pca > 0)
						{
							clps_ok_current_count_pca--;
						}
						
						if (clps_ok_current_count_pca == 0)
						{
							if (clps_ok_detected_pca) clps_ok_detected_pca = false;
						}
						
					}
					else
					{
						if (clps_ok_current_count_pca < CLPS_NCOUNT) clps_ok_current_count_pca++;
						else
						{
							if (!clps_ok_detected_pca)
							{
								logMessage("CLPS-OK Detected by PCA");
								if (jobLog) jobLogMessage("CLPS-OK Detected by PCA");
								clps_ok_detected_pca = true;
							}
						}

						if (clps_current_count_pca > 0) clps_current_count_pca--;
						if (clps_current_count_pca == 0)
						{
							if (clps_detected_pca) clps_detected_pca = false;
						}
					}
				}
				else
				{
					///below the threshold height.
					if (clps_current_count_pca > 0) clps_current_count_pca--;
					if (clps_current_count_pca == 0)
					{
						if (clps_detected_pca) clps_detected_pca = false;
					}
					if (clps_ok_current_count_pca > 0) clps_ok_current_count_pca = 0;
					if (clps_ok_detected_pca) clps_ok_detected_pca = false;
				}
			}
			else if (sprd_landed)
			{
				//release both clps, clps-ok.
				if (clps_current_count_pca > 0) clps_current_count_pca = 0;
				if (clps_detected_pca) clps_detected_pca = false;
				if (clps_ok_current_count_pca > 0)
				{
					clps_ok_current_count_pca--;
				}
				if (clps_ok_current_count_pca == 0)
				{
					if (clps_ok_detected_pca) clps_ok_detected_pca = false;
				}
			}
		}

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[CLPS-Detection-PCA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[CLPS-Detection-PCA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[CLPS-Detection-PCA] Unknown Exception!");
		if (jobLog) jobLogMessage("[CLPS-Detection-PCA] Unknown Exception!");
		return false;
	}
}

bool Position_Detection_PCA(std::string SENSOR_POSITION, bool isLeftSensor, bbx target_cone, pcl::PointCloud<pcl::PointXYZ>::Ptr pcHole, pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone, pcl::PointXYZ& hole_pos, pcl::PointXYZ& cone_pos, std::string save_path = "", bool jobLog = false)
{
	try
	{
		if (pcHole->points.size() > 0)
		{
			//pcl::PointXYZ hole_pos2;
			hole_pos = pc_hole_detection_naive(pcHole, isLeftSensor, save_path);
			logMessage("Detected hole position: (T:" + std::to_string(hole_pos.x) + " , H:" + std::to_string(hole_pos.y) + " , G:" + std::to_string(hole_pos.z) + ")");
			if (jobLog) jobLogMessage("Detected hole position: (T:" + std::to_string(hole_pos.x) + " , H:" + std::to_string(hole_pos.y) + " , G:" + std::to_string(hole_pos.z) + ")");

			/*
			auto t_start = std::chrono::high_resolution_clock::now();
			auto temp = FindCornerCastingCorner_XZPlanes_LowerLeftClosest(pcHole);
			logMessage("Time to FindCornerCastingCorner_XZPlanes_LowerLeftClosest: " +
				std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::high_resolution_clock::now() - t_start).count()) + " msec");

			if (temp.x != std::numeric_limits<float>::quiet_NaN())
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr holeCloud(new pcl::PointCloud <pcl::PointXYZ>);
				holeCloud->points.push_back(temp);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_gpt_hole_pos.pcd", *holeCloud);
			}
			*/
			if (hole_pos.x != -10000)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr holeCloud(new pcl::PointCloud <pcl::PointXYZ>);
				holeCloud->points.push_back(hole_pos);
				if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_pos.pcd", *holeCloud);
			}
		}
		else
		{
			hole_pos = pcl::PointXYZ(-10004, -10004, -10004);
		}

		if (pcCone->points.size() > 0)
		{
			if (target_cone.label == 1)
			{
				cone_pos = pc_cone_detection_naive(pcCone, isLeftSensor, save_path);
				logMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected cone position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_pos.pcd", *coneCloud);
				}
			}
			else
			{
				cone_pos = pc_landed_detection_naive(pcCone, isLeftSensor, save_path);
				logMessage("Detected landed position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");
				if (jobLog) jobLogMessage("Detected landed position: (T:" + std::to_string(cone_pos.x) + " , H:" + std::to_string(cone_pos.y) + " , G:" + std::to_string(cone_pos.z) + ")");

				if (cone_pos.x != -10000)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr coneCloud(new pcl::PointCloud <pcl::PointXYZ>);
					coneCloud->points.push_back(cone_pos);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_landed_pos.pcd", *coneCloud);
				}
			}
		}
		else
		{
			cone_pos = pcl::PointXYZ(-10004, -10004, -10004);
		}

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Position-Detection-PCA] " + std::string(ex.what()));
		if (jobLog) jobLogMessage("[Position-Detection-PCA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Position-Detection-PCA] Unknown Exception!");
		if (jobLog) jobLogMessage("[Position-Detection-PCA] Unknown Exception!");
		return false;
	}
}

bool Pre_Land_Chassis_Position_PCA(bbx target_cone, pcl::PointXYZ cone_pos, pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone, bool jobLog = false)
{
	try
	{
		if (target_cone.prob > 0)
		{
			//initial chassis position.
			if (AVG_CONE_PCA.get_count() < LDO_NCOUNT)
			{
				if (target_cone.prob >= 70 && pcCone->points.size() > 0)
				{
					if (cone_pos.x != -10000)
					{
						AVG_CONE_PCA.update(cone_pos);
					}
				}
			}
			else
			{
				if (!AVG_CONE_PCA.get_set())
				{
					auto bb = AVG_CONE_PCA.get_average();
					logMessage("Pre-Land Chassis Position PCA: (T: " + std::to_string(bb.x) + " , H: " + std::to_string(bb.y) + " , G: " + std::to_string(bb.z));
					if (jobLog) jobLogMessage("Pre-Land Chassis Position PCA: (T: " + std::to_string(bb.x) + " , H: " + std::to_string(bb.y) + " , G: " + std::to_string(bb.z));

					AVG_CONE_PCA.set();
				}
			}
		}
	}
	catch (std::exception& ex)
	{
		logMessage("[Pre-Land-Chassis-Position-PCA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Pre-Land-Chassis-Position-PCA] Unknown Exception!");
		return false;
	}
}

bool Deviation_Output_PCA(std::string SENSOR_POSITION, pcl::PointXYZ hole_pos, pcl::PointXYZ cone_pos, bbx target_cone, IDEAL_POS IP, bool sprd_landed, int& devOut_pca_x, int& devOut_pca_y, int& devOut_pca_z, bool jobLog = false)
{
	try
	{
		bool isLeftSensor = (SENSOR_POSITION == "REAR_LEFT") ? true : false;

		if (hole_pos.x > -10000 && cone_pos.x > -10000)
		{
			//based on TUAS-3DSP.
			int x_offset = 0;
			int z_offset = 0;
			
			if (TZ_Mount_Cycle)
			{
				//Hole - Cone
				if (isLeftSensor) devOut_pca_x = (hole_pos.x - x_offset) - cone_pos.x;
				else devOut_pca_x = (hole_pos.x + x_offset) - cone_pos.x;
				devOut_pca_y = cone_pos.y - hole_pos.y;
				devOut_pca_z = cone_pos.z - (hole_pos.z + z_offset);

				if (sprd_landed || devOut_pca_y > -50)
				{
					if (std::abs(devOut_pca_x) > 50 || std::abs(devOut_pca_z) > 50)
					{
						if (landout_current_count_pca < LDO_NCOUNT)
						{
							landout_current_count_pca++;
						}
						if (landout_current_count_pca >= LDO_NCOUNT)
						{
							if (!landout_detected_pca)
							{
								logMessage("Landout detected by PCA!");
								if (jobLog) jobLogMessage("Landout detected by PCA!");
							}
							landout_detected_pca = true;
						}
					}
					else if (sprd_landed && devOut_pca_y < -50)
					{
						//Detected Cone is above detected hole.
						if (landout_current_count_pca < LDO_NCOUNT)
						{
							landout_current_count_pca++;
						}
						if (landout_current_count_pca >= LDO_NCOUNT)
						{
							if (!landout_detected_pca)
							{
								logMessage("Landout detected by PCA - DevY: " + std::to_string(devOut_pca_y) + "!");
								if (jobLog) jobLogMessage("Landout detected by PCA - DevY: " + std::to_string(devOut_pca_y) + "!");
							}
							landout_detected_pca = true;
						}
					}
					else
					{
						if (landout_current_count_pca > 0) landout_current_count_pca--;
						if (landout_current_count_pca == 0)
						{
							landout_detected_pca = false;
						}
					}
				}
				
				else
				{
					if (landout_current_count_pca > 0) landout_current_count_pca--;
					if (landout_current_count_pca == 0)
					{
						landout_detected_pca = false;
					}
				}
			}
			else if (TZ_Offload_Cycle)
			{
				//Compare landed position with ideal position.
				if (target_cone.label == 2)
				{
					devOut_pca_x = cone_pos.x - IP.LANDED_PCA.x;
					devOut_pca_y = cone_pos.y - IP.LANDED_PCA.y;
					devOut_pca_z = cone_pos.z - IP.LANDED_PCA.z;

					//here we may be able to deduce clps? if something is detected.
				}
				else
				{
					//unloaded cone is selected as target?
					//Cone - Hole
					devOut_pca_x = hole_pos.x - cone_pos.x;
					devOut_pca_y = cone_pos.y - hole_pos.y;
					devOut_pca_z = cone_pos.z - hole_pos.z;
				}
			}
		}
		else if (cone_pos.x != -10000)
		{
			/*
			//No hole, only cone.
			if (TZ_Mount_Cycle)
			{
				devOut_pca_x = cone_pos.x - IP.CONE_PCA.x;
				devOut_pca_y = cone_pos.y - IP.CONE_PCA.y;
				devOut_pca_z = cone_pos.z - IP.CONE_PCA.z;
			}
			*/
		}
		else if (hole_pos.x == -10004 || cone_pos.x == -10004)
		{
			devOut_pca_x = -10004;
			devOut_pca_y = -10004;
			devOut_pca_z = -10004;
		}

		logMessage("PCA Dev Out: (T: " + std::to_string(devOut_pca_x) + " H: " + std::to_string(devOut_pca_y) + " G: " + std::to_string(devOut_pca_z) + ")");
		if (jobLog) jobLogMessage("PCA Dev Out: (T: " + std::to_string(devOut_pca_x) + " H: " + std::to_string(devOut_pca_y) + " G: " + std::to_string(devOut_pca_z) + ")");
		//logMessage("PCA LandOut Detected: " + std::to_string(landout_detected_pca));
		//if (jobLog) jobLogMessage("PCA LandOut Detected PCA: " + std::to_string(landout_detected_pca));

		return true;
	}
	catch (std::exception& ex)
	{
		logMessage("[Deviation-Output-PCA] " + std::string(ex.what()));
		return false;
	}
	catch (...)
	{
		logMessage("[Deviation-Output-PCA] Unknown Exception!");
		return false;
	}
}


ProcessResults ProcessLogic(std::string OP_SENSOR_POS, JobInfo JOB_INFO, const cv::Mat image, pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud, std::string savePath, bool jobLog, bool debugMode, std::string& logLines)
{
	//Local process result output.
	ProcessResults procResults;
	try
	{
		JOB_INFO.print_jobInfo();
		logMessage("Operation Sensor Position: " + OP_SENSOR_POS);

		int _g_data_limit = G_DATA_LIMIT;
		int _t_data_limit = 2000;
		bool _isLeftSensor = false;
		if (OP_SENSOR_POS.find("LEFT") != std::string::npos)
		{
			_t_data_limit = -1 * T_DATA_LIMIT;
			_isLeftSensor = true;
		}
		else { _t_data_limit = T_DATA_LIMIT; }

		cv::Mat res_image = image.clone();

		int _det_count = 0;
		std::vector<rectangle_info> _det_results;
		std::vector<std::vector<bbx>> _det_sorted_objects(class_names.size(), std::vector<bbx>(0));

		auto inference_time = std::chrono::high_resolution_clock::now();
		auto inference_status = yolo_inference(yolo_detector, OP_SENSOR_POS, image, ref(res_image), ref(_det_results), ref(_det_sorted_objects), ref(_det_count), savePath);
		auto inference_Endtime = std::chrono::high_resolution_clock::now();
		auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_Endtime - inference_time);
		logMessage("VA Inference Time= " + std::to_string(inference_duration.count()) + "ms");
		if (jobLog) jobLogMessage("VA Inference Time= " + std::to_string(inference_duration.count()) + "ms");

		//Initialize variables
		std::string _detected_chassisType = "Unknown";
		bool _detected_xt = false, _detected_cst = false, _detected_chassis_type_unknown = false;

		bbx _target_hole, _target_cone, _target_landed, _target_guide;
		int _tCntr_x = -10000, _tCntr_y = -10000, _tCntr_prob = 0;
		int _tCone_x = -10000, _tCone_y = -10000, _tCone_prob = 0;

		int _devOut_x = -10000, _devOut_y = -10000;
		int _devOut_LDO_x = -10000, _devOut_LDO_y = -10000;
		int _devOut_x_mm = -10000, _devOut_y_mm = -10000;

		bool _sprdLanded = false;

		bool _debug_landed_trigger = false;
		bool _usingLDO_Base = false;

		bool _detected_hole = false;

		//PCA Variables
		pcl::PointCloud<pcl::PointXYZ>::Ptr _basePointCloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr _pcHole(new pcl::PointCloud <pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr _pcCone(new pcl::PointCloud <pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr _pcGuide(new pcl::PointCloud <pcl::PointXYZ>);

		pcl::PointXYZ _hole_pos(-10000, -10000, -10000);
		pcl::PointXYZ _cone_pos(-10000, -10000, -10000);

		int _devOut_pca_x = -10000; int _devOut_pca_y = -10000; int _devOut_pca_z = -10000;
		bool _usingPCA_Base = false;

		int _refPCUnderHole = 0, _clps_y_level = 0;
		int _pca_dev_y = 0;

		if (_det_count == 0)
		{
			_detected_xt = false;
			_detected_cst = false;
			_detected_chassis_type_unknown = true;

			logMessage("No acceptable inference results are available. Proceeding to PCA");
			if (jobLog) jobLogMessage("No acceptable inference results are available. Proceeding to PCA");
			
			logMessage("ChassisType: Unknown");
			if (jobLog) jobLogMessage("ChassisType: Unknown");
		}
		else
		{

			auto t_now = std::chrono::high_resolution_clock::now();

			_detected_chassisType = chassis_type_selection_VA(_det_sorted_objects, jobLog);
			_detected_xt = (_detected_chassisType == "XT") ? true : false;
			_detected_cst = (_detected_chassisType == "CST") ? true : false;
			_detected_chassis_type_unknown = (_detected_chassisType == "Unknown") ? true : false;

			Target_Selections_VA(OP_SENSOR_POS, JOB_INFO.jobSize, _det_sorted_objects, ref(_target_hole), ref(_target_cone), ref(_target_guide), ref(_detected_hole));

			Pre_Land_chassis_position_VA(JOB_INFO.isMountCycle, JOB_INFO.isOffloadCycle, _target_cone, _target_hole);

			Deviation_Output_VA(_detected_xt, _detected_cst, _detected_chassis_type_unknown, JOB_INFO.isMountCycle, _target_hole, _target_cone, _target_guide, JOB_INFO.preset, ref(_tCntr_x), ref(_tCntr_y), ref(_tCntr_prob), ref(_tCone_x), ref(_tCone_y), ref(_tCone_prob), ref(_devOut_x), ref(_devOut_y), ref(_devOut_x_mm), ref(_devOut_y_mm), ref(_usingLDO_Base));

			bool _sprdLanded_FallingEdge = false;
			if (debugMode)
			{
				_debug_landed_trigger = Debug_Landed_Trigger(_target_hole, _target_cone);
				_sprdLanded = _debug_landed_trigger;
			}
			else //Live Op.
			{
				if (_sprdLanded != SPRD_Landed && _sprdLanded) 				
				{
					_sprdLanded_FallingEdge = true;
				}
				else
				{
					if (_sprdLanded_FallingEdge) _sprdLanded_FallingEdge = false;
				}

				_sprdLanded = SPRD_Landed;
			}

			if (JOB_INFO.isMountCycle)
			{
				//Avg Cone can be used up until Spreader Landed.
				if (_sprdLanded || (_debug_landed_trigger && debugMode)) bAvg_VA_Usable = false;
			}
			else if (JOB_INFO.isOffloadCycle)
			{
				if (_sprdLanded_FallingEdge || (!_debug_landed_trigger && debugMode)) bAvg_VA_Usable = false;
			}

			LandOut_Detected_VA(JOB_INFO.isMountCycle, _detected_xt, _detected_cst, _target_hole, _target_cone, _sprdLanded, _detected_hole, ref(_devOut_LDO_x), ref(_devOut_LDO_y), _usingLDO_Base, jobLog);

			CLPS_Detection_VA(JOB_INFO.isOffloadCycle, _target_hole, _target_cone, _sprdLanded, jobLog);

			//if no valid VA.
			if (_target_cone.prob == 0)
			{
				if (inferenceFailCounter < 3)
				{
					inferenceFailCounter++;
					if (bAvg_VA_Usable && AVG_CONE.get_set())
					{
						//this is only after VA op is done. --> Result is only used for PCA target.
						_target_cone = AVG_CONE.get_average();
					}
				}
			}
			else //_target_cone detected. 
			{
				if (inferenceFailCounter > 0) inferenceFailCounter--;
			}

			//Write on image for landout_va, clps_va, clps-ok_va.
			if (debugMode)
			{
				if (_sprdLanded) cv::putText(res_image, std::string("LANDED:TRUE"), cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				else cv::putText(res_image, std::string("LANDED:FALSE"), cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

				if (JOB_INFO.isMountCycle)
				{
					//Landout by logic
					if (landout_detected) cv::putText(res_image, std::string("LANDOUT_VA:TRUE"), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
					else cv::putText(res_image, std::string("LANDOUT_VA:FALSE"), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

					if (landok_detected) cv::putText(res_image, std::string("LANDOK_VA:TRUE"), cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
					else cv::putText(res_image, std::string("LANDOK_VA:FALSE"), cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				}
				else
				{
					//CLPS - VA
					if (clps_detected) cv::putText(res_image, std::string("CLPS_VA:TRUE"), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
					else cv::putText(res_image, std::string("CLPS_VA:FALSE"), cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

					//CLPS-OK - VA
					if (clps_ok_detected) cv::putText(res_image, std::string("CLPS-OK_VA:TRUE"), cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
					else cv::putText(res_image, std::string("CLPS-OK_VA:FALSE"), cv::Point(0, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

					auto diff_string = std::to_string(UPPER_DIFF_X) + "," + std::to_string(UPPER_DIFF_Y) + "," + std::to_string(LOWER_DIFF_X) + "," + std::to_string(LOWER_DIFF_Y);
					cv::putText(res_image, std::string("DIFFs(U,L):") + diff_string, cv::Point(0, 110), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				}
			}
			auto proc_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_now).count();
			logMessage("VA Proc Time in ms : " + std::to_string(proc_time_ms) + "ms");
			if (jobLog) jobLogMessage("VA Proc Time in ms : " + std::to_string(proc_time_ms) + "ms");
		}

		logLines += std::to_string(_target_hole.x) + ";" + std::to_string(_target_hole.y) + ";" + std::to_string(_target_hole.w) + ";" + std::to_string(_target_hole.h) + ";" + std::to_string(_target_hole.prob) + ";";
		logLines += std::to_string(_target_cone.label) + ";" + std::to_string(_target_cone.x) + ";" + std::to_string(_target_cone.y) + ";" + std::to_string(_target_cone.w) + ";" + std::to_string(_target_cone.h) + ";" + std::to_string(_target_cone.prob) + ";";
		logLines += std::to_string(_target_guide.x) + ";" + std::to_string(_target_guide.y) + ";" + std::to_string(_target_guide.w) + ";" + std::to_string(_target_guide.h) + ";" + std::to_string(_target_guide.prob) + ";";

		logLines += std::to_string(_devOut_x) + ";" + std::to_string(_devOut_y) + ";";
		logLines += std::to_string(_devOut_x_mm) + ";" + std::to_string(_devOut_y_mm) + ";";
		//2025.03.11
		logLines += std::to_string(_sprdLanded) + ";" + std::to_string(landout_detected) + ";" + std::to_string(landok_detected) + ";" + std::to_string(clps_detected) + ";" + std::to_string(clps_ok_detected) + ";";

		if (savePath != std::string("") && AVG_CONE.get_average().prob > 0 && _target_cone.prob == 0)
		{
			res_image = drawOnImage(res_image, AVG_CONE.get_average(), cv::Scalar(193, 182, 255));
			//cv::imwrite(save_path + "_result.jpg", res_img);
		}

		auto t_start_pca = std::chrono::high_resolution_clock::now();

		//Just copy the point cloud to basePointCloud.
		if (pointCloud == nullptr) logMessage("Input PointCloud is nullptr!");
		else pcl::copyPointCloud(*pointCloud, *_basePointCloud);

		Target_PointCloud_Extraction(_basePointCloud, _isLeftSensor, _target_hole, _target_cone, _target_guide, _sprdLanded, JOB_INFO.preset, _t_data_limit, _g_data_limit, ref(_pcHole), ref(_pcCone), ref(_pcGuide), bAvg_VA_Usable, savePath, jobLog);

		Position_Detection_PCA(OP_SENSOR_POS, _isLeftSensor, _target_cone, _pcHole, _pcCone, ref(_hole_pos), ref(_cone_pos), savePath, jobLog);
		Pre_Land_Chassis_Position_PCA(_target_cone, _cone_pos, _pcCone, jobLog);
		
		/*
		if (AVG_CONE_PCA.get_count() >= LDO_NCOUNT && _cone_pos.x == -10000)
		{
			auto CONE_Base = AVG_CONE_PCA.get_average();
			//if cone is not detected, then use PCA_Base as cone position.
			_cone_pos.x = CONE_Base.x;
			_cone_pos.y = CONE_Base.y;
			_cone_pos.z = CONE_Base.z;
			_usingPCA_Base = true;
		}
		*/
		Deviation_Output_PCA(OP_SENSOR_POS, _hole_pos, _cone_pos, _target_cone, JOB_INFO.preset, _sprdLanded, ref(_devOut_pca_x), ref(_devOut_pca_y), ref(_devOut_pca_z), jobLog);

		logMessage("Dev Out PCA x: " + std::to_string(_devOut_pca_x) + " y: " + std::to_string(_devOut_pca_y) + " z: " + std::to_string(_devOut_pca_z));
		if (jobLog) jobLogMessage("Dev Out PCA x: " + std::to_string(_devOut_pca_x) + " y: " + std::to_string(_devOut_pca_y) + " z: " + std::to_string(_devOut_pca_z));

		if (JOB_INFO.isMountCycle) {
			logMessage("LandOut PCA: " + std::to_string(landout_detected_pca));
			if (jobLog) jobLogMessage("LandOut PCA: " + std::to_string(landout_detected_pca));
		}
		
		if (debugMode) //if (!clps_ok_detected || debugMode)
			CLPS_Detection_PCA(_basePointCloud, _isLeftSensor, _target_hole, _hole_pos, _g_data_limit, _t_data_limit, _detected_hole, _sprdLanded, ref(_refPCUnderHole), ref(_clps_y_level), savePath, jobLog);

		bool _blnSeparated = false;
		if (debugMode) //((!clps_ok_detected && !clps_ok_detected_pca) || debugMode)
			_blnSeparated = CLPS_Detection_PCA_Only(_basePointCloud, _isLeftSensor, ref(_pca_dev_y), savePath);

		auto t_stop_pca = std::chrono::high_resolution_clock::now();
		auto t_dur_pca = std::chrono::duration_cast<std::chrono::milliseconds>(t_stop_pca - t_start_pca).count();
		logMessage("PCA Proc Time in ms: " + std::to_string(t_dur_pca) + "ms");
		if (jobLog) jobLogMessage("PCA Proc Time in ms: " + std::to_string(t_dur_pca) + "ms");


		bCLPS_Detected_Out = (clps_detected); //|| (!clps_ok_detected && clps_detected_pca));
		bCLPS_OK_Detected_Out = (clps_ok_detected);// || clps_ok_detected_pca || clps_ok_detected_pca_only);

		bLandout_Detected_Out = (landout_detected);// || landout_detected_pca);
		bLandOK_Detected_Out = (landok_detected);// || landok_detected_pca);

		logLines += std::to_string(_pcHole->points.size()) + ";" + std::to_string(_pcCone->points.size()) + ";" + std::to_string(_pcGuide->points.size()) + ";";
		logLines += std::to_string(_hole_pos.x) + ";" + std::to_string(_hole_pos.y) + ";" + std::to_string(_hole_pos.z) + ";";
		logLines += std::to_string(_cone_pos.x) + ";" + std::to_string(_cone_pos.y) + ";" + std::to_string(_cone_pos.z) + ";";

		auto CONE_AVG = AVG_CONE_PCA.get_average();
		logLines += std::to_string(CONE_AVG.x) + ";" + std::to_string(CONE_AVG.y) + ";" + std::to_string(CONE_AVG.z) + ";";

		logLines += std::to_string(_devOut_pca_x) + ";" + std::to_string(_devOut_pca_y) + ";" + std::to_string(_devOut_pca_z) + ";";
		logLines += std::to_string(landout_detected_pca) + ";" + std::to_string(landok_detected_pca) + ";" + std::to_string(clps_detected_pca) + ";" + std::to_string(clps_ok_detected_pca) + ";";

		logLines += std::to_string(_refPCUnderHole) + ";" + std::to_string(LDO_Current_Count) + ";" + std::to_string(LandOK_Current_Count) + ";" + std::to_string(landout_current_count_pca) + ";" + std::to_string(landok_current_count_pca) + ";";
		logLines += std::to_string(CLPS_Current_Count) + ";" + std::to_string(clps_current_count_pca) + ";";
		logLines += std::to_string(clps_ok_current_count_pca) + ";" + std::to_string(_clps_y_level) + ";";

		logLines += std::to_string(_blnSeparated) + ";" + std::to_string(_pca_dev_y) + ";";

		logLines += std::to_string(bCLPS_Detected_Out) + ";" + std::to_string(bCLPS_OK_Detected_Out) + ";";
		logLines += std::to_string(bLandout_Detected_Out) + ";" + std::to_string(bLandOK_Detected_Out);

		if (JOB_INFO.isOffloadCycle)
		{
			logMessage("CLPS , CLPS-OK Status: " + std::to_string(bCLPS_Detected_Out) + " , " + std::to_string(bCLPS_OK_Detected_Out));
			if (jobLog) jobLogMessage("CLPS , CLPS-OK Status: " + std::to_string(bCLPS_Detected_Out) + " , " + std::to_string(bCLPS_OK_Detected_Out));
		}
		if (JOB_INFO.isMountCycle)
		{
			logMessage("Landout, Land-OK Status: " + std::to_string(bLandout_Detected_Out) + " , " + std::to_string(bLandOK_Detected_Out));
			if (jobLog) jobLogMessage("Landout, Land-OK Status: " + std::to_string(bLandout_Detected_Out) + " , " + std::to_string(bLandOK_Detected_Out));
		}

		if (debugMode)
		{
			if (JOB_INFO.isOffloadCycle)
			{
				//CLPS-OK - VA
				if (clps_detected_pca) cv::putText(res_image, std::string("CLPS_PCA:TRUE"), cv::Point(0, 140), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				else cv::putText(res_image, std::string("CLPS_PCA:FALSE"), cv::Point(0, 140), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

				if (clps_ok_detected_pca) cv::putText(res_image, std::string("CLPS-OK_PCA:TRUE"), cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				else cv::putText(res_image, std::string("CLPS-OK_PCA:FALSE"), cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

				if (clps_ok_detected_pca_only) cv::putText(res_image, std::string("CLPS-SEP_PCA:TRUE : ") + std::to_string(_pca_dev_y), cv::Point(0, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				else cv::putText(res_image, std::string("CLPS-SEP_PCA:FALSE : ") + std::to_string(_pca_dev_y), cv::Point(0, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

				//draw line here.
				if (_clps_y_level > 0) cv::line(res_image, cv::Point(0, _clps_y_level), cv::Point(512, _clps_y_level), cv::Scalar(0, 0, 255));
				cv::putText(res_image, std::string("PCunder: ") + std::to_string(_refPCUnderHole), cv::Point(0, 230), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			}
			else if (JOB_INFO.isMountCycle)
			{
				cv::putText(res_image, std::string("DEV_PCA: ") + std::to_string(_devOut_pca_x) + " y: " + std::to_string(_devOut_pca_y) + " z: " + std::to_string(_devOut_pca_z), cv::Point(0, 140), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

				if (landout_detected_pca) cv::putText(res_image, std::string("LANDOUT_PCA:TRUE"), cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
				else cv::putText(res_image, std::string("LANDOUT_PCA:FALSE"), cv::Point(0, 170), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
			}

			if (savePath != std::string("")) cv::imwrite(savePath + "_result.jpg", res_image);
		}
		//Update to Current Processing Result.

		//procResults
		procResults.bDetected_XT = _detected_xt;
		procResults.bDetected_CST = _detected_cst;
		procResults.bDetected_Unknown = _detected_chassis_type_unknown;

		procResults.VA_Chassis_Type = _detected_chassisType;

		procResults.bDetected_Container = _detected_hole;
		procResults.bDetected_Chassis = (_detected_xt || _detected_cst);

		procResults.VA_Target_Container = _target_hole;
		procResults.VA_Target_Chassis = _target_cone;
		//procResults.PCA_Target_Guide = _target_guide;

		procResults.VA_Devout_X = _devOut_x;
		procResults.VA_Devout_Y = _devOut_y;

		procResults.VA_isCLPS = clps_detected;
		procResults.VA_isCLPS_OK = clps_ok_detected;
		procResults.VA_isLandout = landout_detected;
		procResults.VA_isLandOK = landok_detected;

		procResults.PCA_Target_Chassis = _cone_pos;
		procResults.PCA_Target_Container = _hole_pos;
		//procResults.PCA_Target_Guide = _guide_pos;

		procResults.PCA_Devout_X = _devOut_pca_x;
		procResults.PCA_Devout_Y = _devOut_pca_y;
		procResults.PCA_Devout_Z = _devOut_pca_z;

		procResults.PCA_isCLPS = clps_detected_pca;
		procResults.PCA_isCLPS_OK = clps_ok_detected_pca;
		procResults.PCA_isLandout = landout_detected_pca;
		procResults.PCA_isLandOK = landok_detected_pca;

		procResults.bClpsDetected = bCLPS_Detected_Out;
		procResults.bClpsOkDetected = bCLPS_OK_Detected_Out;
		procResults.bLandOutDetected = bLandout_Detected_Out;
		procResults.bLandOKDetected = bLandOK_Detected_Out;

		return procResults;
	}
	catch (std::exception& ex)
	{
		logMessage("[Process-Logic] " + std::string(ex.what()));
		return procResults;
	}
	catch (...)
	{
		logMessage("[Process-Logic] Unknown Exception!");
		return procResults;
	}
}

void processingThread()
{
	try
	{
		//__try
		//{
			logMessage("Processing Thread Activated!");

			while (proc_running.load())
			{
				std::unique_lock<std::mutex> lock(mutex_processing);
				bool res = cond_processing.wait_for(lock,
					std::chrono::seconds(3600),
					[]() { return flag; });

				if (res)
				{
					
					logMessage("Enabled process! Current Position : " + CURRENT_SENSOR_POSITION);
					jobLogMessage("Enabled process! Current Position : " + CURRENT_SENSOR_POSITION);

					//data log file.
					auto logHeader = std::string("Filename;Hole_x;Hole_y;Hole_w;Hole_h;Hole_prob;XT_index;XT_x;XT_y;XT_w;XT_h;XT_prob;Guide_x;Guide_y;Guide_w;Guide_y;Guide_prob;devout_x;devout_y;devout_x_mm;devout_y_mm;Landed_Trigger;Landout_Detected;LandOK_Detected;CLPS_Detected;CLPS_OK;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;cone_avg_x;cone_avg_y;cone_avg_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca;landok_pca;clps_pca;clpsOk_pca;pcUnderHole;LDO_VA_Count;LDO_PCA_Count;LandOK_VA_Count;LandOK_PCA_Count;CLPS_VA_Count;CLPS_PCA_Count;CLPSOK_PCA_Count;CLPS_Y_LEVEL;CLPS_PCA_ONLY_SEP;SEP_DEV_Y;CLPS_OUT;CLPS_OK_OUT;LandOut_OUT;LandOK_Out");

					ofstream Simfile;
					std::string sim_result_txt = job_result_folder_name + "/" + job_name + "_res.txt";
					Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
					if (!Simfile.is_open())
					{
						logMessage("Failed to open file?");
					}
					if (Simfile.is_open())
					{
						Simfile << logHeader + "\n";
						Simfile.close();
					}

					int frame_index = 0;

					//printf("Process starting.\n");
					auto check_interval = 100;
					auto last_dt_checked = std::chrono::system_clock::now();

					bool dataStackReady = false;

					logMessage("Process Setup Complete!");
					while (enable_process)
					{
						auto get_dt = std::chrono::system_clock::now();
						auto wait_dur = std::chrono::duration_cast<std::chrono::milliseconds>(get_dt - last_dt_checked).count();
						if (wait_dur >= check_interval)
						{
							if (dataStack.Stack_Ready())
							{
								dataStackReady = true;
								frame_index++;
								

								last_dt_checked = get_dt;

								cv::Mat img;
								pcl::PointCloud<pcl::PointXYZ>::Ptr pcPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
								pcl::PointCloud<pcl::PointXYZ> pointCloud;
								std::vector<uint16_t> distMap = {};
								std::chrono::system_clock::time_point last_ts, ts;

								//logMessage("Retrieving stack data...");
								auto r_status = dataStack.Retrieve_Stack(ref(img), ref(pointCloud), ref(distMap), ref(ts));
								//if (r_status) logMessage("Stack data retrieved successfully!");
								if (!r_status)
								{
									logMessage("Failed to retrieve stack data!");
									std::this_thread::sleep_for(std::chrono::milliseconds(5)); //single frame capture time.
									continue;
								}

								
								//Convert visionary::pointcloud to pcl::pointcloud
								if (pcPointCloud == NULL)
								{
									pcPointCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
									logMessage("Created new pcl::PointCloud<pcl::PointXYZ> instance.");
								}
								pcPointCloud->points.resize(pointCloud.size());
								std::copy(pointCloud.begin(), pointCloud.end(), pcPointCloud->points.begin());

								auto res_img = img.clone();
								if (r_status && last_ts < ts)
								{
									last_ts = ts;
									std::string log_lines = time_as_name(last_ts) + ";";

									if (save_trigger_by_landed || save_trigger_by_TWL_Locked)
									{
										//Draw and push to queue.
										//res_img = drawOnImage(res_img, det_results);

										std::string msg = save_trigger_by_TWL_Locked ? "TWL" : "TWUL";
										std::string msg2 = save_trigger_by_landed ? "LANDED" : "LANDOFF";
										msg = msg + std::string("_") + msg2;

										auto dataTup = std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::vector<uint16_t>, bool, std::string, std::chrono::system_clock::time_point>(img, res_img, *pcPointCloud, distMap, true, msg, ts);
										tsq.push(dataTup);

										if (save_trigger_by_landed) save_trigger_by_landed = false;
										if (save_trigger_by_TWL_Locked) save_trigger_by_TWL_Locked = false;
									}
									else if (enable_logging)
									{

										//save data to queue with copied data.
										cv::Mat saveMat = img.clone();
										pcl::PointCloud<pcl::PointXYZ> savePointCloud;
										pcl::copyPointCloud(pointCloud, savePointCloud);

										auto dataTup = std::tuple<cv::Mat, cv::Mat, pcl::PointCloud<pcl::PointXYZ>, std::vector<uint16_t>, bool, std::string, std::chrono::system_clock::time_point>(saveMat, cv::Mat(), savePointCloud, distMap, false, "", last_frame_get_time);
										tsq.push(dataTup);
									}

									current_process_results = ProcessLogic(CURRENT_SENSOR_POSITION, current_job_info, img, pcPointCloud, std::string(""), true, false, log_lines);
										
									Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
									if (!Simfile.is_open())
									{
										logMessage("Failed to open file?");
									}
									if (Simfile.is_open())
									{
										Simfile << log_lines + "\n";
										Simfile.close();
									}
								}
								
							}
							else
							{
								if (dataStackReady)
								{
									dataStackReady = false;
									logMessage("Data Stack is NOT ready for processing. Waiting for data...");
								}
								std::this_thread::sleep_for(std::chrono::milliseconds(100));
							}
						}
						else
						{
							auto sleep_time = check_interval - wait_dur;
							std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
						}
					}

					logMessage("Process thread is back to blocked state...");
				}
				else
				{
					//Timeout triggered awake.
					logMessage("Timeout triggered awake, returning to wait_for...");
				}
			}

			logMessage("Prcessing Thread Deactivated!");
		//}
		//__except (EXCEPTION_EXECUTE_HANDLER)
		//{
		//	logMessage("Processing thread exception SEH (Access Violation, Stack Overflow caught!");
		//}
	}
	catch (std::exception& ex)
	{
		logMessage("Processing thread exception: " + std::string(ex.what()));
	}
	catch (...)
	{
		logMessage("Processing thread has been terminated due to an unknown exception!");
	}
}

void OfflineDebugBatchProcessingThread()
{
	logMessage("Debug Batch Job on " + DEBUG_BATCH_ROOT_DIR + " to be processed and saved to " + DEBUG_BATCH_SAVE_DIR);

	//Get list of sub directories given ROOT DIR.
	auto jobDirectories = ListSubDirectories(DEBUG_BATCH_ROOT_DIR);

	bool savePLY = true;

	auto logHeader_offload_batch = std::string("Total_Job;OK3;OK-VA,OK-PCA;OK-VA,OK-PCA-ONLY;OK-PCA,OK-PCA-ONLY;OK-VA;OK-PCA;OK-PCA-ONLY;OK-N;CLPS-VA,CLPS-PCA;CLPS-VA;CLPS-PCA;CLPS-N");

	//Per job: directory name, then results.
	//For batch: total jobs and each count.
	int total_jobs = 0, OK_VA = 0, OK_VA_OK_PCA = 0, OK_VA_OK_PCA_ONLY = 0, OK3 = 0, OK_PCA = 0, OK_PCA_OK_PCA_ONLY = 0, OK_PCA_ONLY = 0, OK_N = 0;
	int CLPS_VA = 0, CLPS_VA_CLPS_PCA = 0, CLPS_PCA = 0, CLPS_N = 0;

	ofstream Simfile_batch;
	std::string sim_batch_result_txt = DEBUG_BATCH_ROOT_DIR + "/Offload_batch_results.txt";
	Simfile_batch.open(sim_batch_result_txt.c_str(), ios::out | ios::app);
	if (!Simfile_batch.is_open())
	{
		logMessage("Failed to open file?");
	}
	if (Simfile_batch.is_open())
	{
		Simfile_batch << logHeader_offload_batch + "\n";
		Simfile_batch.close();
	}

	for (const auto& jobDir : jobDirectories)
	{
		TZ_Mount_Cycle = false;
		TZ_Offload_Cycle = false;

		reset_jobVariables();

		logMessage("Processing: " + jobDir);

		//load .jpg, .ply files for processing
		auto image_filePath = jobDir + "/" + std::string("Image");
		auto depth_filePath = jobDir + "/" + std::string("Depth");
		//auto dist_filePath = jobDir + "/" + std::string("DistMap");

		std::filesystem::path pathObj = std::filesystem::path(jobDir).lexically_normal();
		std::string lastDirectory = pathObj.filename().string();

		auto image_files = getAllFiles(image_filePath, ".jpg");
		auto depth_files = getAllFiles(depth_filePath, ".ply");
		if (depth_files.size() == 0) depth_files = getAllFiles(depth_filePath, ".pcd");
		//auto dist_files = getAllFiles(dist_filePath, ".zstd");

		createDirectory_ifexists(DEBUG_BATCH_SAVE_DIR);
		auto save_file_path = DEBUG_BATCH_SAVE_DIR + "/" + lastDirectory;
		auto folder_created = createDirectory_ifexists(save_file_path);

		if (!folder_created)
		{
			logMessage("Skipping already existing folder: " + save_file_path);
			continue;
		}

		logMessage("Saving to " + save_file_path);

		auto logHeader = std::string("Filename;Hole_x;Hole_y;Hole_w;Hole_h;Hole_prob;XT_index;XT_x;XT_y;XT_w;XT_h;XT_prob;Guide_x;Guide_y;Guide_w;Guide_y;Guide_prob;devout_x;devout_y;devout_x_mm;devout_y_mm;Landed_Trigger;Landout_Detected;LandOK_Detected;CLPS_Detected;CLPS_OK;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;cone_avg_x;cone_avg_y;cone_avg_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca;landok_pca;clps_pca;clpsOk_pca;pcUnderHole;LDO_VA_Count;LDO_PCA_Count;LandOK_VA_Count;LandOK_PCA_Count;CLPS_VA_Count;CLPS_PCA_Count;CLPSOK_PCA_Count;CLPS_Y_LEVEL;CLPS_PCA_ONLY_SEP;SEP_DEV_Y;CLPS_OUT;CLPS_OK_OUT;LandOut_OUT;LandOK_Out");

		ofstream Simfile;
		std::string sim_result_txt = save_file_path + "/" + lastDirectory + "_res.txt";
		Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
		if (!Simfile.is_open())
		{
			logMessage("Failed to open file?");
		}
		if (Simfile.is_open())
		{
			Simfile << logHeader + "\n";
			Simfile.close();
		}

		//Get job information from the folder name.
		auto info_temp = split(lastDirectory, '_');
		if (info_temp.size() > 3)
		{
			if (info_temp[3] == "Mount")
			{
				TZ_Mount_Cycle = true;
				TZ_Offload_Cycle = false;
			}
			else if (info_temp[3] == "Offload")
			{
				TZ_Offload_Cycle = true;
				TZ_Mount_Cycle = false;
			}
			else
			{
				TZ_Mount_Cycle = false;
				TZ_Offload_Cycle = false;
			}
		}
		int sprd_size = 40;
		if (info_temp.size() > 7)
		{
			if (info_temp[7] == "SPRD45")
			{
				sprd_size = 45;
			}
			else if (info_temp[7] == "SRPD20")
			{
				sprd_size = 20;
			}
		}

		IDEAL_POS IP;
		int g_data_limit = G_DATA_LIMIT;
		int t_data_limit = 2000;

		bool isLeftSensor = false;
		if (DEBUG_SENSOR_POSITION.find("LEFT") != std::string::npos)
		{
			IP = L_Pos;
			t_data_limit = -1 * T_DATA_LIMIT;
			isLeftSensor = true;
		}
		else
		{
			IP = R_Pos;
			t_data_limit = T_DATA_LIMIT;
		}

		JobInfo debugJobInfo;
		debugJobInfo.set(TZ_Mount_Cycle, TZ_Offload_Cycle, "Center", "XT", sprd_size, sprd_size, IP);

		for (int i = 0; i < image_files.size(); i++)
		{
			logMessage("Processing Cycle Start!");

			std::filesystem::path pathObj(image_files.at(i));
			// Get the filename with extension
			std::string filename = pathObj.stem().string();

			std::string log_lines = filename + ";";

			//folder for each file
			auto save_current_file_path = save_file_path + "/" + filename;
			createDirectory_ifexists(save_current_file_path);

			std::string save_path = save_current_file_path + "/" + filename;
			
			//load image
			cv::Mat image = cv::imread(image_files.at(i));
			cv::Mat res_image;
			if (image.empty())
			{
				logMessage("Couldn't read file by JPG " + image_files.at(i));
				continue;
			}

			//load ply
			bool loadPCD_fail = false;
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcData = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
			if (pcl::io::loadPCDFile<pcl::PointXYZ>(depth_files.at(i), *pcData) == -1)
			{
				std::cout << std::string("Couldn't read file by PLY ") << std::endl;
				loadPCD_fail = true;
				//return false;
			}
			//pcl::PointCloud<pcl::PointXYZ>::Ptr pcData = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
			if (pcl::io::loadPLYFile<pcl::PointXYZ>(depth_files.at(i), *pcData) == -1)
			{
				std::cout << std::string("Couldn't read file by PLY ") << std::endl;
				//loadPCD_fail = true;
				//return false;
			}

			
			auto pointCloud = makePCL_PointCloud(*pcData, DEBUG_CONVERT_PCL_RANGE);
			//if (pointCloud->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(save_current_file_path + "/" + filename + "_base_converted.pcd", *pointCloud);
			if (pointCloud->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(save_current_file_path + "/" + filename + "_base_converted.pcd", *pointCloud);
			//trying loading zstd?
			//auto tDistMap = load_u16_zstd(dist_files.at(i));
			//logMessage("dist size: " + std::to_string(tDistMap.size()));

			//std::vector<PointXYZ> pointCloud2;
			//pDataHandler->generatePointCloud(pointCloud);
			//pDataHandler->generatePointCloud2(tDistMap, pointCloud2);

			//auto pclPointCloud = makePCL_PointCloud(pointCloud2);
			//if (pclPointCloud->points.size() > 0) pcl::io::savePCDFileBinaryCompressed(save_current_file_path + "/" + filename + "_base_distMap_converted.pcd", *pclPointCloud);

			auto procRes = ProcessLogic(DEBUG_SENSOR_POSITION, debugJobInfo, image, pointCloud, save_path, false, true, log_lines);


			Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
			if (!Simfile.is_open())
			{
				logMessage("Failed to open file?");
			}
			if (Simfile.is_open())
			{
				Simfile << log_lines + "\n";
				Simfile.close();
			}
		}

		total_jobs++;

		//offload batch test
		bool bOK_VA = false, bOK_VA_OK_PCA = false, bOK_VA_OK_PCA_ONLY = false, bOK3 = false, bOK_PCA = false, bOK_PCA_OK_PCA_ONLY = false, bOK_PCA_ONLY = false, bOK_N = false;
		bool bCLPS_VA = false, bCLPS_VA_CLPS_PCA = false, bCLPS_PCA = false, bCLPS_N = false;

		//CLPS results
		if (clps_detected && clps_detected_pca) { bCLPS_VA_CLPS_PCA = true; CLPS_VA_CLPS_PCA++; }
		else if (clps_detected) { bCLPS_VA = true; CLPS_VA++; }
		else if (clps_detected_pca) { bCLPS_PCA = true; CLPS_PCA++; }
		else { bCLPS_N = true; CLPS_N++;  }

		//CLPS-OK results
		if (clps_ok_detected && clps_ok_detected_pca && clps_ok_detected_pca_only) { bOK3 = true; OK3++; }
		if (clps_ok_detected && clps_ok_detected_pca) { bOK_VA_OK_PCA = true; OK_VA_OK_PCA++; }
		if (clps_ok_detected && clps_ok_detected_pca_only) { bOK_VA_OK_PCA_ONLY = true; OK_VA_OK_PCA_ONLY++; }
		if (clps_ok_detected_pca && clps_ok_detected_pca_only) { bOK_PCA_OK_PCA_ONLY = true; OK_PCA_OK_PCA_ONLY++; }
		
		if (clps_ok_detected) { bOK_VA = true; OK_VA++; }
		if (clps_ok_detected_pca) { bOK_PCA = true; OK_PCA++; }
		if (clps_ok_detected_pca_only) { bOK_PCA_ONLY = true; OK_PCA_ONLY++; }
		if (!clps_ok_detected && !clps_ok_detected_pca && !clps_ok_detected_pca_only) { bOK_N = true; OK_N++; }

		//auto logHeader_offload_batch = std::string("Total_Job;OK3;OK-VA,OK-PCA;OK-VA,OK-PCA-ONLY;OK-PCA,OK-PCA-ONLY;OK-VA;OK-PCA;OK-PCA-ONLY;OK-N;CLPS-VA,CLPS-PCA;CLPS-VA;CLPS-PCA;CLPS-N");

		std::string batch_line = lastDirectory + ";" + std::to_string(bOK3) + ";" + std::to_string(bOK_VA_OK_PCA) + ";" + std::to_string(bOK_VA_OK_PCA_ONLY) + ";" + std::to_string(bOK_PCA_OK_PCA_ONLY) + ";"
			+ std::to_string(bOK_VA) + ";" + std::to_string(bOK_PCA) + ";" + std::to_string(bOK_PCA_ONLY) + ";" + std::to_string(bOK_N);
		batch_line += ";" + std::to_string(bCLPS_VA_CLPS_PCA) + ";" + std::to_string(bCLPS_VA) + ";" + std::to_string(bCLPS_PCA) + ";" + std::to_string(bCLPS_N);

		//ofstream Simfile_batch;
		//std::string sim_batch_result_txt = DEBUG_BATCH_ROOT_DIR + "/Offload_batch_results.txt";
		Simfile_batch.open(sim_batch_result_txt.c_str(), ios::out | ios::app);
		if (!Simfile_batch.is_open())
		{
			logMessage("Failed to open file?");
		}
		if (Simfile_batch.is_open())
		{
			Simfile_batch << batch_line + "\n";
			Simfile_batch.close();
		}
	}

	std::string batch_line = std::to_string(total_jobs) + ";" + std::to_string(OK3) + ";" + std::to_string(OK_VA_OK_PCA) + ";" + std::to_string(OK_VA_OK_PCA_ONLY) + ";" + std::to_string(OK_PCA_OK_PCA_ONLY) + ";"
		+ std::to_string(OK_VA) + ";" + std::to_string(OK_PCA) + ";" + std::to_string(OK_PCA_ONLY) + ";" + std::to_string(OK_N);
	batch_line += ";" + std::to_string(CLPS_VA_CLPS_PCA) + ";" + std::to_string(CLPS_VA) + ";" + std::to_string(CLPS_PCA) + ";" + std::to_string(CLPS_N);

	//ofstream Simfile_batch;
	//std::string sim_batch_result_txt = DEBUG_BATCH_ROOT_DIR + "/Offload_batch_results.txt";
	Simfile_batch.open(sim_batch_result_txt.c_str(), ios::out | ios::app);
	if (!Simfile_batch.is_open())
	{
		logMessage("Failed to open file?");
	}
	if (Simfile_batch.is_open())
	{
		Simfile_batch << batch_line + "\n";
		Simfile_batch.close();
	}
}

pcl::PointXYZ pc_hole_detection_zedx(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ hole_position(10000, -10000, 10000);
	try
	{
		//Get Y slices :: 20 height slices.
		//Cornercastings height = usually 120mm.
		int y_slice_height = 20;

		int minY, maxY;
		get_max_min_y(pcInput, ref(maxY), ref(minY));

		std::vector<pcl::PointXYZ> target_points = {};

		int i = 0;
		for (int cy = minY; cy < maxY; cy += y_slice_height)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(pcInput, cy, cy + y_slice_height, "y", slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > 75)
				{
					int minZ = -10000, maxZ = -10000;
					get_max_min_z(slice_, ref(maxZ), ref(minZ));

					//from minZ slice, get minX.
					int minX = -10000, maxX = -10000;
					get_max_min_x(slice_, ref(maxX), ref(minX));

					//Get maxZ and minX.
					auto targetPt = pcl::PointXYZ(minX, cy, maxZ);
					if (!isLeftSide) targetPt.x = maxX;

					target_points.push_back(targetPt);
					pcl::PointCloud<pcl::PointXYZ>::Ptr tPt(new pcl::PointCloud<pcl::PointXYZ>);
					tPt->points.push_back(targetPt);
					if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_hole_targetPt_i_" + std::to_string(i) + ".pcd", *tPt);
				}
			}
			i++;
		}

		//based on target points.
		//average x and z while get min y.
		float avg_x = 0.0f, avg_z = 0.0f;
		for (auto& it : target_points)
		{
			if (it.y > hole_position.y) hole_position.y = it.y;
			avg_x += it.x;
			avg_z += it.z;
		}
		avg_x = avg_x / target_points.size();
		avg_z = avg_z / target_points.size();

		hole_position.x = avg_x;
		hole_position.z = avg_z;
	}

	catch (std::exception& ex)
	{
		logMessage("[PCA_HOLE] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_HOLE] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return hole_position;
}

//PointCloud :: Cone Detection (XT)?
pcl::PointXYZ pc_cone_detection_zedx(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcInput, bool isLeftSide, std::string save_path)
{
	pcl::PointXYZ cone_position(10000, 10000, 10000);
	try
	{
		//slice through y direction with slice height = 20.

		//slice from top to bottom.
		//look for most points then append one slice above as well.

		int y_slice_height = 20;
		int max_slice_index = 6;

		int minY, maxY;
		get_max_min_y(pcInput, ref(maxY), ref(minY));

		pcl::PointCloud<pcl::PointXYZ>::Ptr prev_y_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpy_above_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpy_slice(new pcl::PointCloud<pcl::PointXYZ>);

		int i = 0;
		for (int cy = minY; cy < maxY; cy += y_slice_height)
		{
			if (i > max_slice_index) break;

			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(pcInput, cy, cy + y_slice_height, "y", slice_);
			
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_y_i_" + std::to_string(i) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > mpy_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpy_slice);
					pcl::copyPointCloud(*prev_y_slice, *mpy_above_slice);
				}
			}

			if (i > 0) pcl::copyPointCloud(*slice_, *prev_y_slice);
			i++;
		}

		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_maxYslice_pc_" + std::to_string(mpy_slice->points.size()) + ".pcd", *mpy_slice);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_above_maxYslice_pc_" + std::to_string(mpy_above_slice->points.size()) + ".pcd", *mpy_above_slice);

		//Combine two slices.
		pcl::PointCloud<pcl::PointXYZ>::Ptr combined_slice(new pcl::PointCloud<pcl::PointXYZ>);
		*combined_slice = *mpy_slice + *mpy_above_slice;

		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_combined_pc_" + std::to_string(combined_slice->points.size()) + ".pcd", *combined_slice);

		//slice through X, Y to get most points slice (30).
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpx_slice(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mpz_slice(new pcl::PointCloud<pcl::PointXYZ>);

		int mp_slice_width = 30;
		int minX, maxX, minZ, maxZ;
		get_max_min_x(combined_slice, ref(maxX), ref(minX));
		get_max_min_z(combined_slice, ref(maxZ), ref(minZ));

		int ix = 0;
		for (int cx = minX; cx < maxX; cx += 10)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(combined_slice, cx, cx + mp_slice_width, "x", slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_x_i_" + std::to_string(ix) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > mpx_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpx_slice);
				}
			}
			ix++;
		}

		int iz = 0;
		for (int cz = minZ; cz < maxZ; cz += 10)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr slice_(new pcl::PointCloud<pcl::PointXYZ>);
			auto val = pc_passThrough(combined_slice, cz, cz + mp_slice_width, "z", slice_);
			if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_slice_z_i_" + std::to_string(iz) + "pc_" + std::to_string(slice_->points.size()) + ".pcd", *slice_);
			if (val)
			{
				if (slice_->points.size() > mpz_slice->points.size())
				{
					pcl::copyPointCloud(*slice_, *mpz_slice);
				}
			}
			iz++;
		}

		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_mpx_pc_" + std::to_string(mpx_slice->points.size()) + ".pcd", *mpx_slice);
		if (save_path != std::string("")) pcl::io::savePCDFileBinaryCompressed(save_path + "_cone_mpz_pc_" + std::to_string(mpz_slice->points.size()) + ".pcd", *mpz_slice);

		int combined_minY, combined_maxY;
		get_max_min_y(combined_slice, ref(combined_maxY), ref(combined_minY));

		//get minZ from mpx_slice.
		int mpx_minZ, mpx_maxZ;
		get_max_min_z(mpx_slice, ref(mpx_maxZ), ref(mpx_minZ));
		
		cone_position.z = mpx_maxZ;

		//get minX from mpz_slice for LEFT SENSOR.
		//get maxX from mpz_slice for RIGHT SENSOR.
		int mpz_minX, mpz_maxX;
		get_max_min_x(mpz_slice, ref(mpz_maxX), ref(mpz_minX));

		if (isLeftSide) cone_position.x = mpz_minX;
		else cone_position.x = mpz_maxX;

		cone_position.y = combined_minY + (combined_maxY - combined_minY) / 2;
	}
	catch (std::exception& ex)
	{
		logMessage("[PCA_CONE] " + std::string(ex.what()));
		return pcl::PointXYZ(-10000, -10000, -10000);
	}
	catch (...)
	{
		logMessage("[PCA_CONE] Unknown Exception!");
		return pcl::PointXYZ(-10000, -10000, -10000);
	}

	return cone_position;
}

void ZedX_Processing_Test()
{
	logMessage("ZedX Processing Test!");
	//Get Hole and Cone extracted data,
	std:string data_path = "I:/LTSP/20250317_LTSP_ZEDX_40FT_normal_batch_Results/FLZ";

	//load .jpg, .ply files for processing
	auto hole_filePath = data_path + "/" + std::string("Hole_Original");
	logMessage(hole_filePath);
	auto cone_filePath = data_path + "/" + std::string("Cone_Original");

	std::filesystem::path pathObj = std::filesystem::path(data_path).lexically_normal();
	std::string lastDirectory = pathObj.filename().string();

	auto hole_files = getAllFiles(hole_filePath, ".pcd");
	logMessage(std::to_string(hole_files.size()));
	auto cone_files = getAllFiles(cone_filePath, ".pcd");

	std::string save_path = data_path;

	auto logHeader = std::string("Filename;Hole_pc;Cone_pc;Guide_pc;hole_x;hole_y;hole_z;cone_x;cone_y;cone_z;devout_pca_x;devout_pca_y;devout_pca_z;landout_pca");

	ofstream Simfile;
	std::string sim_result_txt = save_path + "/Hole/BatchResult.txt";
	Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
	if (!Simfile.is_open())
	{
		logMessage("Failed to open file?");
	}
	if (Simfile.is_open())
	{
		Simfile << logHeader + "\n";
		Simfile.close();
	}

	int fileLen = hole_files.size();
	for (int i = 0; i < fileLen; i++)
	{
		std::string hole_file = hole_files[i];
		std::filesystem::path pathObj(hole_file);
		// Get the filename with extension
		std::string filename = pathObj.stem().string();
		std::string current_hole_save_path = save_path + "/Hole/" + filename;
		std::string current_cone_save_path = save_path + "/Cone/" + filename;
		std::string log_lines = filename + ";";
		//load data.
		pcl::PointCloud<pcl::PointXYZ>::Ptr pcHole(new pcl::PointCloud <pcl::PointXYZ>);
		if (pcl::io::loadPLYFile<pcl::PointXYZ>(hole_file, *pcHole) == -1)
		{
			std::cout << std::string("Couldn't read file by PLY ") << std::endl;
			//return false;
		}
		std::string cone_file = cone_files[i];
		pcl::PointCloud<pcl::PointXYZ>::Ptr pcCone(new pcl::PointCloud <pcl::PointXYZ>);
		if (pcl::io::loadPLYFile<pcl::PointXYZ>(cone_file, *pcCone) == -1)
		{
			std::cout << std::string("Couldn't read file by PLY ") << std::endl;
			//return false;
		}

		//base data
		pcl::io::savePCDFileBinaryCompressed(current_hole_save_path + "_pcHole.pcd", *pcHole);
		//base data
		pcl::io::savePCDFileBinaryCompressed(current_cone_save_path + "_pcCone.pcd", *pcCone);

		//noisefilter
		pcl::PointCloud<pcl::PointXYZ>::Ptr nf_hole(new pcl::PointCloud<pcl::PointXYZ>);
		nf_hole = open3d_NoiseFilter(pcHole, 30, 1.0);
		pcl::io::savePCDFileBinaryCompressed(current_hole_save_path + "_pcNFHole.pcd", *nf_hole);

		pcl::PointCloud<pcl::PointXYZ>::Ptr nf_cone(new pcl::PointCloud<pcl::PointXYZ>);
		nf_cone = open3d_NoiseFilter(pcCone, 30, 1.0);
		pcl::io::savePCDFileBinaryCompressed(current_cone_save_path + "_pcNFCole.pcd", *nf_cone);

		log_lines += std::to_string(pcHole->points.size()) + ";" + std::to_string(pcCone->points.size()) + "; 0; ";
		
		//hole processing.
		auto hole_pos = pc_hole_detection_zedx(nf_hole, true, current_hole_save_path);
		pcl::PointCloud<pcl::PointXYZ>::Ptr ptHole(new pcl::PointCloud <pcl::PointXYZ>);
		ptHole->points.push_back(hole_pos);
		pcl::io::savePCDFileBinaryCompressed(current_hole_save_path + "_hole_pos.pcd", *ptHole);
		log_lines += std::to_string(hole_pos.x) + ";" + std::to_string(hole_pos.y) + ";" + std::to_string(hole_pos.z) + ";";
		
		//cone processing.
		auto cone_pos = pc_cone_detection_zedx(nf_cone, true, current_cone_save_path);
		pcl::PointCloud<pcl::PointXYZ>::Ptr ptCone(new pcl::PointCloud <pcl::PointXYZ>);
		ptCone->points.push_back(cone_pos);
		pcl::io::savePCDFileBinaryCompressed(current_cone_save_path + "_cone_pos.pcd", *ptCone);
		log_lines += std::to_string(cone_pos.x) + ";" + std::to_string(cone_pos.y) + ";" + std::to_string(cone_pos.z) + ";";

		log_lines += "0;0;0;0;";
		Simfile.open(sim_result_txt.c_str(), ios::out | ios::app);
		if (!Simfile.is_open())
		{
			logMessage("Failed to open file?");
		}
		if (Simfile.is_open())
		{
			Simfile << log_lines + "\n";
			Simfile.close();
		}
	}
}

void handleClient(SOCKET clientSocket) {
	logMessage("Client connected. Sending initial message...");

	char buffer[DEFAULT_RECVLEN] = { 0 };
	while (socket_running.load()) 
	{
		int bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0);
		if (bytesReceived <= 0) {
			logMessage("Client disconnected. Waiting for new client...");
			break;
		}
		//buffer[bytesReceived] = '\0';

		parseCommand(buffer);

		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		
		char sendbuf[DEFAULT_SENDLEN] = {};
		makeCommand(ref(sendbuf));

		int bytesSent = send(clientSocket, sendbuf, DEFAULT_SENDLEN, 0);
		if (bytesSent <= 0)
		{
			logMessage("Sending data failed!");
			break;
		}
	}
	closesocket(clientSocket);
}

//2025.02.06 Updated socket server
int start_server() {
	WSADATA wsaData;
	SOCKET serverSocket;
	struct sockaddr_in serverAddr;
	int addrlen = sizeof(serverAddr);

	// Initialize Winsock
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		std::cerr << "WSAStartup failed!\n";
		logMessage("WSAStartup failed!");
		return 1;
	}

	// Create socket
	serverSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (serverSocket == INVALID_SOCKET) {
		std::cerr << "Socket creation failed!\n";
		logMessage("Socket creation failed!");
		WSACleanup();
		return 1;
	}

	BOOL on = TRUE;
	if (setsockopt(serverSocket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
		reinterpret_cast<const char*>(&on), sizeof(on)) == SOCKET_ERROR) {
		std::cerr << "Socket reuse port failed!\n";
		logMessage("Socket reuse port failed!");
		WSACleanup();
		return 1;
	}

	// Bind socket
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_addr.s_addr = INADDR_ANY;
	serverAddr.sin_port = htons(SOCKET_PORT);

	if (::bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
		std::cerr << "Bind failed!\n";
		logMessage("Bind failed!");
		closesocket(serverSocket);
		WSACleanup();
		return 1;
	}

	// Start listening
	if (listen(serverSocket, SOMAXCONN) == SOCKET_ERROR) {
		std::cerr << "Listen failed!\n";
		logMessage("Listen failed");
		closesocket(serverSocket);
		WSACleanup();
		return 1;
	}

	logMessage("Server listening on port " + std::to_string(SOCKET_PORT) + "...");

	while (socket_running.load()) 
	{
		try
		{
			struct sockaddr_in clientAddr;
			int clientSize = sizeof(clientAddr);
			SOCKET clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientSize);

			if (clientSocket == INVALID_SOCKET)
			{
				logMessage("Accept failed!");
				std::cerr << "Accept failed!\n";
				continue;
			}

			// Handle client in a separate thread
			std::thread clientThread(handleClient, clientSocket);

			clientThread.join();  // Wait for the thread to finish

			//clientThread.detach();  // Let it run independently
			logMessage("Waiting for new client...");
		}
		catch (std::exception& ex)
		{
			logMessage("[SERVER] " + std::string(ex.what()));
		}
		catch (...)
		{
			logMessage("[SERVER] Unknown Exception!");
		}
	}

	logMessage("Terminating Socket Server");
	closesocket(serverSocket);
	WSACleanup();
	return 0;
}

void socketClient(const std::string& server_ip, int server_port)
{
	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);  // Init Winsock

	while (socket_running) {
		SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (sock == INVALID_SOCKET) {
			std::cerr << "Socket creation failed\n";
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}

		sockaddr_in server_addr{};
		server_addr.sin_family = AF_INET;
		server_addr.sin_port = htons(server_port);
		inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);

		std::cout << "Attempting to connect to " << server_ip << ":" << server_port << "...\n";
		if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
			std::cerr << "Connection failed: " << WSAGetLastError() << "\n";
			closesocket(sock);
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}

		std::cout << "Connected to server!\n";

		//initial send?
		char sendbuf_init[DEFAULT_SENDLEN] = { 0 };
		//makeCommand(ref(sendbuf));
		int bytesSent_init = send(sock, sendbuf_init, DEFAULT_SENDLEN, 0);

		// Sample loop: Read from server
		char buffer[DEFAULT_RECVLEN] = { 0 };
		while (socket_running) {
			int bytesReceived = recv(sock, buffer, sizeof(buffer), 0);
			if (bytesReceived <= 0) {
				std::cerr << "Disconnected from server\n";
				break;
			}
			//buffer[bytesReceived] = '\0';
			std::string received_msg(buffer);
			//std::cout << "Received: " << buffer << "\n";

			parseCommand(buffer);

			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			char sendbuf[DEFAULT_SENDLEN] = { 0 };
			makeCommand(ref(sendbuf));

			int bytesSent = send(sock, sendbuf, DEFAULT_SENDLEN, 0);
			if (bytesSent <= 0)
			{
				logMessage("Sending data failed! " + std::to_string(bytesSent));
				//std::cerr << "Send failed: " << strerror(errno) << "\n";
				break;
			}

			//std::cout << "Sent: " << sendbuf << "\n";
		}

		closesocket(sock);
		logMessage("Connection closed. Retrying in 2 seconds...");
		std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait before retry
	}

	WSACleanup();
}

bool runStreamingDemo(const char ipAddress[], unsigned short dataPort, uint32_t numberOfFrames, bool executeExtTrigger)
{
	using namespace visionary;

	// Generate Visionary instance
	auto pDataHandler = std::make_shared<VisionaryTMiniData>();
	VisionaryDataStream dataStream(pDataHandler);
	VisionaryControl visionaryControl;

	std::printf("Made samples.\n");
	std::cout << ipAddress << " ;;; " << dataPort << std::endl;

	//-----------------------------------------------
	// Connect to devices data stream 
	if (!dataStream.open(ipAddress, htons(dataPort)))
	{
		std::printf("Failed to open data stream connection to device.\n");
		return false;   // connection failed
	}

	std::printf("Data stream opened.\n");

	//-----------------------------------------------
	// Connect to devices control channel
	if (!visionaryControl.open(VisionaryControl::ProtocolType::COLA_2, ipAddress, 5000/*ms*/))
	{
		std::printf("Failed to open control connection to device.\n");
		return false;   // connection failed
	}

	std::printf("Control stream opened.\n");


	//-----------------------------------------------
	// read Device Ident
	std::printf("DeviceIdent: '%s'\n", visionaryControl.getDeviceIdent().c_str());

	//-----------------------------------------------
	// Login as authorized client
	if (visionaryControl.login(IAuthentication::UserLevel::AUTHORIZED_CLIENT, "CLIENT"))
	{
		//-----------------------------------------------
		// An example of reading an writing device parameters is shown here.
		// Use the "SOPAS Communication Interface Description" PDF to determine data types for other variables
		//-----------------------------------------------
		// Set enDepthMask parameter to false

		std::printf("Setting enDepthMask to false\n");
		CoLaCommand setEnDepthMaskCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "enDepthMask").parameterBool(false).build();
		CoLaCommand setEnDepthMaskResponse = visionaryControl.sendCommand(setEnDepthMaskCommand);
		if (setEnDepthMaskResponse.getError() == CoLaError::OK)
		{
			std::printf("Successfully set enDepthMask to false\n");
		}


		//-----------------------------------------------
		// Read humidity parameter
		CoLaCommand getHumidity = CoLaParameterWriter(CoLaCommandType::READ_VARIABLE, "humidity").build();
		CoLaCommand humidityResponse = visionaryControl.sendCommand(getHumidity);
		const double humidity = CoLaParameterReader(humidityResponse).readLReal();
		std::printf("Read humidity = %f\n", humidity);

		//-----------------------------------------------
		// Read info messages variable
		CoLaCommand getMessagesCommand = CoLaParameterWriter(CoLaCommandType::READ_VARIABLE, "MSinfo").build();
		CoLaCommand messagesResponse = visionaryControl.sendCommand(getMessagesCommand);

		//-----------------------------------------------
	}

	{
		CoLaCommand setEnDepthMaskCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "enDepthMask").parameterBool(true).build();
		CoLaCommand setEnDepthMaskResponse = visionaryControl.sendCommand(setEnDepthMaskCommand);
		if (setEnDepthMaskResponse.getError() != CoLaError::OK)
		{
			std::printf("Failed to set enDepthMask to true\n");
		}
	}

	//-----------------------------------------------
	// Logout from device after reading variables.
	if (!visionaryControl.logout())
	{
		std::printf("Failed to logout\n");
	}

	//-----------------------------------------------
	// Stop image acquisition (works always, also when already stopped)
	visionaryControl.stopAcquisition();

	//-----------------------------------------------
	// Capture a single frame
	visionaryControl.stepAcquisition();
	if (dataStream.getNextFrame())
	{
		std::printf("Frame received through step called, frame #%d, timestamp: %u \n", pDataHandler->getFrameNum(), pDataHandler->getTimestampMS());

		//-----------------------------------------------
		// Convert data to a point cloud
		std::vector<PointXYZ> pointCloud;
		pDataHandler->generatePointCloud(pointCloud);
		pDataHandler->transformPointCloud(pointCloud);



		//-----------------------------------------------
		// Write point cloud to PLY
		const char plyFilePath[] = "VisionaryT.pcd";
		std::printf("Writing frame to %s\n", plyFilePath);
		PointCloudPlyWriter::WriteFormatPLY(plyFilePath, pointCloud, pDataHandler->getIntensityMap(), true);
		std::printf("Finished writing frame to %s\n", plyFilePath);
	}

	//-----------------------------------------------
	// Start image acquisiton and continously receive frames
	visionaryControl.startAcquisition();
	for (uint32_t i = 0; i < numberOfFrames; i++)
	{
		if (!dataStream.getNextFrame())
		{
			continue;     // No valid frame received
		}
		std::printf("Frame received in continuous mode, frame #%d \n", pDataHandler->getFrameNum());
		std::vector<uint16_t> intensityMap = pDataHandler->getIntensityMap();
		std::vector<uint16_t> distanceMap = pDataHandler->getDistanceMap();
		std::vector<uint16_t> stateMap = pDataHandler->getStateMap();
	}

	//-----------------------------------------------
	// This part of the sample code is skipped by default because not every user has a working IO trigger hardware available. 
	// If you want to execute it set variable "executeExtTrigger" in main function to "true".
	if (executeExtTrigger)
	{
		// Capture single frames with external trigger
		// NOTE: This part of the sample only works if you have a working rising egde signal on IO1 which triggers an image!
		std::printf("\n=== Starting external trigger example: \n");
		// Login as authorized client
		if (visionaryControl.login(IAuthentication::UserLevel::AUTHORIZED_CLIENT, "CLIENT"))
		{
			// Set frontendMode to STOP (= 1)
			std::printf("Setting frontendMode to STOP (= 1)\n");
			CoLaCommand setFrontendModeCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "frontendMode").parameterUSInt(1).build();
			CoLaCommand setFrontendModeResponse = visionaryControl.sendCommand(setFrontendModeCommand);
			if (setFrontendModeResponse.getError() != CoLaError::OK)
			{
				std::printf("Failed to set frontendMode to STOP (= 1)\n");
			}

			// Set INOUT1_Function to Trigger (= 7)
			std::printf("Setting DIO1Fnc to Trigger (= 7)\n");
			CoLaCommand setDIO1FncCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "DIO1Fnc").parameterUSInt(7).build();
			CoLaCommand setDIO1FncResponse = visionaryControl.sendCommand(setDIO1FncCommand);
			if (setDIO1FncResponse.getError() != CoLaError::OK)
			{
				std::printf("Failed to set DIO1Fnc to Trigger (= 7)\n");
			}

			// Set INOUT2_Function to TriggerBusy (= 23)
			std::printf("Setting DIO2Fnc to TriggerBusy (= 23)\n");
			CoLaCommand setDIO2FncCommand = CoLaParameterWriter(CoLaCommandType::WRITE_VARIABLE, "DIO2Fnc").parameterUSInt(23).build();
			CoLaCommand setDIO2FncResponse = visionaryControl.sendCommand(setDIO2FncCommand);
			if (setDIO2FncResponse.getError() != CoLaError::OK)
			{
				std::printf("Failed to set DIO2Fnc to TriggerBusy (= 23)\n");
			}
		}

		// Re-Connect to device data stream (make sure there are no old images in the pipeline)
		dataStream.close();
		std::this_thread::sleep_for(std::chrono::seconds(1)); // This short deelay is necessary to not have any old frames in the pipeline.
		if (!dataStream.open(ipAddress, htons(dataPort)))
		{
			std::printf("Failed to open data stream connection to device.\n");
			return false;   // connection failed
		}

		std::printf("Please enable trigger on IO1 to receive an image: \n");
		long long startTime = std::chrono::system_clock::now().time_since_epoch().count();
		long long timeNow = startTime;

		// Limited time loop for receiving hardware trigger signals on IO1 pin
		bool frameReceived = false;
		long long triggerTimeOut = 100000000; // 10 sec = 100 000 000
		while ((timeNow - startTime) <= triggerTimeOut) {
			// Read variable IOValue
			CoLaCommand getIOValue = CoLaParameterWriter(CoLaCommandType::READ_VARIABLE, "IOValue").build();
			CoLaCommand IOValueResponse = visionaryControl.sendCommand(getIOValue);
			CoLaParameterReader IOValues(IOValueResponse);
			const int8_t IOValue1 = IOValues.readSInt();
			const int8_t IOValue2 = IOValues.readSInt(); // We need the IOValue of IO2 from the V3SIOsState struct
			std::printf("Read TriggerBusy = %d\n", IOValue2);

			// Receive the next frame
			if (IOValue2 == 0)
			{
				if (dataStream.getNextFrame())
				{
					std::printf("Frame received in external trigger mode, frame #%d \n", pDataHandler->getFrameNum());
					frameReceived = true;
				}
				timeNow = std::chrono::system_clock::now().time_since_epoch().count();
			}
		}

		if (frameReceived == false)
		{
			std::printf("TIMEOUT: No trigger signal received on IO1 within %.2f seconds!\n", (float)triggerTimeOut / 10000000);
		}
	}
	//-----------------------------------------------

	visionaryControl.close();
	dataStream.close();
	return true;
}

int rtsp_stream_receiver() {
	// Replace with your RTSP stream URL
	//"rtsp://admin:seoho098@192.168.0.163:554/Streaming/channels/101"
	//"rtsp://192.168.15.230:554/live/0"
	std::string rtsp_url = "rtsp://admin:seoho098@192.168.0.163:554";
	cv::VideoCapture capture(rtsp_url, cv::CAP_FFMPEG);

	if (!capture.isOpened()) {
		std::cout << "Error: Cannot open the RTSP stream." << std::endl;
		return -1;
	}

	cv::Mat frame;
	while (true) {
		// Grab a frame
		bool isFrameGrabbed = capture.grab();
		if (!isFrameGrabbed) {
			std::cout << "Error: Cannot grab frame from RTSP stream." << std::endl;
			break;
		}

		// Retrieve and process the frame
		capture.retrieve(frame);

		// Display the frame
		cv::imshow("RTSP Stream", frame);

		// Exit the loop on pressing 'q'
		if (cv::waitKey(30) == 'q') {
			break;
		}
	}

	capture.release();
	cv::destroyAllWindows();
	return 0;
}

std::string Utf16ToUtf8(const std::wstring& w) {
	int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), w.size(), nullptr, 0, nullptr, nullptr);
	std::string s(len, '\0');
	WideCharToMultiByte(CP_UTF8, 0, w.data(), w.size(), &s[0], len, nullptr, nullptr);
	return s;
}
static std::wstring Utf8ToWstring(const std::string& s) {
	if (s.empty()) return L"";
	int size_needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
	std::wstring w(size_needed, 0);
	MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], size_needed);
	return w;
}

struct Detection2 { cv::Rect box; int cls; float score; };

namespace fs = std::filesystem;
std::string getParent(const std::string& pathStr) {
	fs::path p(pathStr);
	return p.parent_path().string();
}
int main(int argc, char* argv[])
{
	wchar_t buf[MAX_PATH];
	DWORD sz = GetModuleFileNameW(nullptr, buf, MAX_PATH);
	if (sz > 0 && sz < MAX_PATH) {
		std::wstring exePath(buf);
		app_path = Utf16ToUtf8(exePath);
	}

	app_path = getParent(app_path);

	HANDLE mutex = CreateMutexA(nullptr, TRUE, app_path.c_str());

	if (GetLastError() == ERROR_ALREADY_EXISTS) {
		std::cerr << "Another instance is already running.\n";
		return 1;
	}

	SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);

	printf("Visionary T-Mini Logging App\n");

	parseAppName(app_path);

	appName = appID + "_v" + program_version;

	std::cout << "App: " << appName << "\n";

	std::thread logThread(logWriterThread);

	logMessage(appName + " Starting!");
	logMessage(app_path);

	signal(SIGINT, my_handler);
	//std::set_terminate(unHandledExceptionHandler);

	parseINI(app_path);
	parseProcessINI(app_path);
	parseIPINI(app_path);

	//YOLOv11
	bool useGPU = true;

	std::string modelPath = app_path + "/" + onnx_model_path;
	std::string labelsPath = app_path + "/INI/classes.txt";

	std::cout << "Initializing YOLOv11 detector with model: " << modelPath << std::endl;
	std::cout << "Classes file: " << labelsPath << std::endl;
	std::cout << "Using confidence threshold: " << confThreshold << ", IoU threshold: " << iouThreshold << std::endl;

	// read model 
	std::cout << "Loading model and labels..." << std::endl;

	//YOLO11Detector detector (modelPath, labelsPath, useGPU);

	yolo_detector.initialize(modelPath, labelsPath, useGPU);
	model_loaded = true;

	std::string imagePath = app_path + "/sample.jpg";
	cv::Mat frame = cv::imread(imagePath);
	cv::Mat res_frame = frame.clone();

	// Perform detection with the updated thresholds
	std::vector<Detection> detections = yolo_detector.detect(frame, confThreshold, iouThreshold);

	model_initialized = (detections.size() > 0) ? true : false;

	if (DEBUG_WITH_FILES || DEBUG_BATCH_JOB)
	{
		//test();

		if (DEBUG_BATCH_JOB) OfflineDebugBatchProcessingThread();

		//std::thread rtspT(rtsp_stream_receiver);
		//rtspT.join();
		//if (DEBUG_BATCH_JOB) ZedX_Processing_Test();
	}
	else if (MODE_DEBUG)
	{
		logMessage("Debug Mode running.. Testing for onnx.");

		//CURRENT_SENSOR_POSITION = "REAR_LEFT";

		DEBUG_IMG_PATH = DEBUG_SAMPLE_JOB + "/Image";
		DEBUG_PLY_PATH = DEBUG_SAMPLE_JOB + "/Depth";

		DEBUG_IMG_FILES = getAllFiles(DEBUG_IMG_PATH, ".jpg");
		DEBUG_PLY_FILES = getAllFiles(DEBUG_PLY_PATH, ".pcd");

		DEBUG_MAX_INDEX = (DEBUG_IMG_FILES.size() < DEBUG_PLY_FILES.size()) ? DEBUG_IMG_FILES.size() : DEBUG_PLY_FILES.size();
		DEBUG_CURRENT_INDEX = 0;

		/*
		demo_img = cv::imread("20250302_041440_568_TMini_Image.jpg", cv::IMREAD_COLOR);

		pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud = std::make_shared<pcl::PointCloud <pcl::PointXYZ>>();
		if (pcl::io::loadPLYFile<pcl::PointXYZ>("20250302_041440_568_TMini_Depth.pcd", *pointCloud) == -1)
		{
			std::cout << std::string("Couldn't read file by PLY ") << std::endl;
			//return false;
		}
		
		demo_ply->points.resize(pointCloud->points.size());

		std::transform(pointCloud->points.begin(), pointCloud->points.end(), demo_ply->points.begin(),
			[](pcl::PointXYZ val) { return pcl::PointXYZ(val.x * 1000, val.y * 1000, val.z * 1000); });
		*/
		//std::thread sckT(start_server);
		std::thread joblogThread(jobLogWriterThread);

		std::thread sckTMini_Setup(thread_tmini_control);
		std::thread sckTMini_Data_Stream(thread_tmini_data_stream);

		//need to specify and enable connection.
		//current_lane_ip = IP_ADDRESSES[0];
		
		//trigger tmini setup and stream.
		//tmini_ctrl_flag = true;
		//cond_tmini_ctrl.notify_one();

		//std::thread tProc(processingThread, std::ref(detector));
		std::thread tSave(data_save_thread);

		//tProc.join();
		joblogThread.join();
		sckTMini_Setup.join();
		sckTMini_Data_Stream.join();
		tSave.join();
		//sckT.join();
	}
	else
	{
		std::thread sckT(start_server);
		//std::thread sckClientThread(socketClient, SOCKET_IP, SOCKET_PORT);

		//Visionary T Mini Streaming setup.
		std::thread sckTMini_Setup(thread_tmini_control);
		std::thread sckTMini_Data_Stream(thread_tmini_data_stream);

		std::thread joblogThread(jobLogWriterThread);

		std::thread tProc(processingThread);

		std::thread tSave(data_save_thread);

		if (tProc.joinable()) tProc.join();
		
		job_log_running.store(false);
		cvJobLog.notify_one();
		if (joblogThread.joinable()) joblogThread.join();
		
		if (sckT.joinable()) sckT.join();
		
		if (trigger_terminate)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			if (sckTMini_Setup.joinable())
			{
				logMessage("Program Just Shutting Down...");

				log_running.store(false);
				cvLog.notify_one();
				logThread.join();

				//std::cout << "Press any key to continue...";
				//system("pause"); // Windows only
				//std::exit(-1);
				return 0;
			}
		}

		if (sckTMini_Setup.joinable()) sckTMini_Setup.join();
		if (sckTMini_Data_Stream.joinable()) sckTMini_Data_Stream.join();
		if (tSave.joinable()) tSave.join();
		//sckClientThread.join();
	}

	logMessage("Program Shutting Down...");

	log_running.store(false);
	cvLog.notify_one();
	logThread.join();

	//std::cout << "Press any key to continue...";
	//system("pause"); // Windows only

	return 1;
}
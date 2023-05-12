#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// 读取点云数据
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <regex>  
// 用于滤波
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
// 用于降采样
#include <pcl/filters/voxel_grid.h>
// 用于计算法向量
#include <pcl/features/normal_3d.h>
// 用于搜索
#include <pcl/search/search.h>
// 用于区域生长
#include <pcl/segmentation/region_growing.h>

// 用于构造Bounding Box
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/common/norms.h>

// 用于投影点云到平面
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

// 用于凹凸多边形
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>

// 用于RANSAC循环分割
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// 用于计算线段相交
#include <pcl/common/impl/intersections.hpp>

/*读取点云数据*/
void readPointCloud(std::string PointCloudFileName,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

/*回调函数 交互选点输出坐标*/
void pp_callback(const pcl::visualization::PointPickingEvent& event, 
    void* viewer_void);

/*统计滤波*/
void statisticalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud);

/*距离滤波*/
void distanceFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud);

/*降采样*/ 
void downSample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud);

/*计算法向量*/ 
void computeNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, 
    pcl::PointCloud<pcl::Normal>::Ptr outputNormal);

/*法向量滤波*/ 
void normalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::Normal>::Ptr inputNormal, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud, 
    pcl::PointCloud<pcl::Normal>::Ptr outputNormal);

/*区域生长*/ 
void useRegionGrow(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::Normal>::Ptr inputNormal, 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, 
    std::vector <pcl::PointIndices> *clusters);

/*计算修正主成分方向与法向量方向*/ 
void computePCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, 
    Eigen::Matrix3f* eigenVectorsPCA, 
    Eigen::Vector4f* pcaCentroid);

//根据索引提取所需点云，indices为索引，negative为false时提取索引所示点云，为true提取剩余点云（因为调用的是滤波器，滤掉true和false，所以提取的点逻辑相反
pcl::PointCloud<pcl::PointXYZ>::Ptr filterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointIndices::Ptr indices,
    bool negative,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);

void Seg_RansacLine(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointIndices::Ptr inliers,
    pcl::ModelCoefficients::Ptr coefficients,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outliers,
    double distance_threshold);

/*提取一条直线并存储*/
bool extractLineByRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left,
    std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines,
    std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines);

/*提取所有直线*/
void allLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input,
    std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines,
    std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines);

/*求线的交点*/
void getIntersecPoints(std::vector<pcl::PointXYZ> *IntersecPoints,
    std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines,
    std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines);

/*根据两点求直线的参数*/
void getLineCoefficients(pcl::PointXYZ pt1,
    pcl::PointXYZ pt2,
    Eigen::VectorXf& line_coeffs);

/*求水平线和多边形的一条边作为线段来说是否有交点（而非作为直线）*/
bool lineSegmentIntersection(const Eigen::VectorXf& height_line_coeffs,
    const Eigen::VectorXf& polygon_line_coeffs,
    Eigen::Vector4f& intersect_point,
    pcl::PointXYZ pt1,
    pcl::PointXYZ pt2);

/*
void getLineCoefficients(pcl::PointXYZ pt1,
    pcl::PointXYZ pt2,
    pcl::ModelCoefficients& line_coeffs);
*/

/*指定一个高度，求改高度线与凸多边的各边的交点*/
void heightWithConvexIntersection(double height,
    double min_x,
    double max_x,
    pcl::PointCloud<pcl::PointXYZ>::Ptr Polygon_points,
    std::vector<Eigen::Vector4f>& Intersect_points);

/*接收一个二维的凸多边形，以及将他变换回三维空间的变换矩阵，返回一条二维的路径，与在三维空间的还原路径*/
void getPathPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points,
    Eigen::Matrix4f restoreTransform,
    std::vector<Eigen::Vector4f>& path_points,
    std::vector<Eigen::Vector4f>& restored_path_points);

/*根据投影到平面上的分割墙面点云，获得墙面的角点*/
void getCornerPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected,
    Eigen::Matrix4f restoreTransform,
    std::vector<Eigen::Vector4f>& corner_points,
    std::vector<Eigen::Vector4f>& restored_corner_points);

/*点类型转换*/
void PclPointXYZ2EigenVector4f(pcl::PointXYZ& pt_pcl,
    Eigen::Vector4f& pt_eigen);

void EigenVector4f2PclPointXYZ(Eigen::Vector4f& pt_eigen,
    pcl::PointXYZ& pt_pcl);

// 基于PCA计算BB
//void getBBbyPCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, pcl::PointXYZ *min_pt, pcl::PointXYZ *max_pt, Eigen::Quaternionf *bboxQuaternion, Eigen::Vector3f *bboxTransform);

# include <stack>
# include <queue>
// 边界关键点提取,cloud_boundary输入原始边界点，boundary_results输出简化后的边界点
void LineBoun(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary, pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_results);

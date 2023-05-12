#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// ��ȡ��������
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <regex>  
// �����˲�
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
// ���ڽ�����
#include <pcl/filters/voxel_grid.h>
// ���ڼ��㷨����
#include <pcl/features/normal_3d.h>
// ��������
#include <pcl/search/search.h>
// ������������
#include <pcl/segmentation/region_growing.h>

// ���ڹ���Bounding Box
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/common/norms.h>

// ����ͶӰ���Ƶ�ƽ��
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

// ���ڰ�͹�����
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>

// ����RANSACѭ���ָ�
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// ���ڼ����߶��ཻ
#include <pcl/common/impl/intersections.hpp>

/*��ȡ��������*/
void readPointCloud(std::string PointCloudFileName,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

/*�ص����� ����ѡ���������*/
void pp_callback(const pcl::visualization::PointPickingEvent& event, 
    void* viewer_void);

/*ͳ���˲�*/
void statisticalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud);

/*�����˲�*/
void distanceFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud);

/*������*/ 
void downSample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud);

/*���㷨����*/ 
void computeNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, 
    pcl::PointCloud<pcl::Normal>::Ptr outputNormal);

/*�������˲�*/ 
void normalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::Normal>::Ptr inputNormal, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud, 
    pcl::PointCloud<pcl::Normal>::Ptr outputNormal);

/*��������*/ 
void useRegionGrow(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointCloud<pcl::Normal>::Ptr inputNormal, 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, 
    std::vector <pcl::PointIndices> *clusters);

/*�����������ɷַ����뷨��������*/ 
void computePCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, 
    Eigen::Matrix3f* eigenVectorsPCA, 
    Eigen::Vector4f* pcaCentroid);

//����������ȡ������ƣ�indicesΪ������negativeΪfalseʱ��ȡ������ʾ���ƣ�Ϊtrue��ȡʣ����ƣ���Ϊ���õ����˲������˵�true��false��������ȡ�ĵ��߼��෴
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

/*��ȡһ��ֱ�߲��洢*/
bool extractLineByRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left,
    std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines,
    std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines);

/*��ȡ����ֱ��*/
void allLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input,
    std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines,
    std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines);

/*���ߵĽ���*/
void getIntersecPoints(std::vector<pcl::PointXYZ> *IntersecPoints,
    std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines,
    std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines);

/*����������ֱ�ߵĲ���*/
void getLineCoefficients(pcl::PointXYZ pt1,
    pcl::PointXYZ pt2,
    Eigen::VectorXf& line_coeffs);

/*��ˮƽ�ߺͶ���ε�һ������Ϊ�߶���˵�Ƿ��н��㣨������Ϊֱ�ߣ�*/
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

/*ָ��һ���߶ȣ���ĸ߶�����͹��ߵĸ��ߵĽ���*/
void heightWithConvexIntersection(double height,
    double min_x,
    double max_x,
    pcl::PointCloud<pcl::PointXYZ>::Ptr Polygon_points,
    std::vector<Eigen::Vector4f>& Intersect_points);

/*����һ����ά��͹����Σ��Լ������任����ά�ռ�ı任���󣬷���һ����ά��·����������ά�ռ�Ļ�ԭ·��*/
void getPathPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points,
    Eigen::Matrix4f restoreTransform,
    std::vector<Eigen::Vector4f>& path_points,
    std::vector<Eigen::Vector4f>& restored_path_points);

/*����ͶӰ��ƽ���ϵķָ�ǽ����ƣ����ǽ��Ľǵ�*/
void getCornerPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected,
    Eigen::Matrix4f restoreTransform,
    std::vector<Eigen::Vector4f>& corner_points,
    std::vector<Eigen::Vector4f>& restored_corner_points);

/*������ת��*/
void PclPointXYZ2EigenVector4f(pcl::PointXYZ& pt_pcl,
    Eigen::Vector4f& pt_eigen);

void EigenVector4f2PclPointXYZ(Eigen::Vector4f& pt_eigen,
    pcl::PointXYZ& pt_pcl);

// ����PCA����BB
//void getBBbyPCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, pcl::PointXYZ *min_pt, pcl::PointXYZ *max_pt, Eigen::Quaternionf *bboxQuaternion, Eigen::Vector3f *bboxTransform);

# include <stack>
# include <queue>
// �߽�ؼ�����ȡ,cloud_boundary����ԭʼ�߽�㣬boundary_results����򻯺�ı߽��
void LineBoun(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary, pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_results);

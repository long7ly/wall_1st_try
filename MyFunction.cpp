#include "MyFunction.h"

/*读取点云数据*/
void readPointCloud(std::string PointCloudFileName,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    std::regex pattern_pcd("^.*\.pcd$");
    std::regex pattern_ply("^.*\.ply$");
    std::smatch match;

    if (std::regex_match(PointCloudFileName, match, pattern_pcd))
    {
        std::cout << "PointCloudFileName ends with .pcd." << std::endl;
        pcl::io::loadPCDFile<pcl::PointXYZ>(PointCloudFileName, *cloud);
    }
    else if(std::regex_match(PointCloudFileName, match, pattern_ply))
    {
        std::cout << "PointCloudFileName ends with .ply." << std::endl;
        pcl::io::loadPLYFile<pcl::PointXYZ>(PointCloudFileName, *cloud);
    }
    else
    {
        std::cout << "PointCloudFileName should end with .pcd or .ply." << std::endl;
    }
	std::cout << "Loaded " << cloud->size() << " data points from " << PointCloudFileName << std::endl;
    
    return;
}

/*回调函数 交互选点输出坐标*/
void pp_callback(const pcl::visualization::PointPickingEvent& event,
    void* viewer_void)
{
    std::cout << "Picking event active" << std::endl;
    if (event.getPointIndex() != -1)
    {
        float x, y, z;
        event.getPoint(x, y, z);
        std::cout << x << "; " << y << "; " << z << std::endl;
    }
}

/*统计滤波*/
void statisticalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(inputCloud);
    sor.setMeanK(10);
    sor.setStddevMulThresh(1);
    sor.filter(*outputCloud);
    std::cout << "has " << outputCloud->size() << " points after Statistical Filter" << std::endl;
}

/*距离滤波*/
void distanceFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud)
{
    pcl::PassThrough<pcl::PointXYZ> XYZpass;
    XYZpass.setInputCloud(inputCloud);
    XYZpass.setFilterFieldName("z");
    XYZpass.setFilterLimits(0.2, 4);
    XYZpass.filter(*outputCloud);
    std::cout << "has " << outputCloud->size() << " points after XYZ Filter" << std::endl;
}

/*降采样*/
void downSample(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud)
{
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(inputCloud);
    //vg.setLeafSize(0.005f, 0.005f, 0.005f);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    //vg.setLeafSize(0.05f, 0.05f, 0.05f);
    vg.filter(*outputCloud);

    std::cout << "Downsampled cloud contains " << outputCloud->size() << " data points" << std::endl;
}

/*计算法向量*/
void computeNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree,
    pcl::PointCloud<pcl::Normal>::Ptr outputNormal)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(inputCloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(100);
    //ne.setKSearch(50);
    ne.compute(*outputNormal);

    std::cout << "Computed " << outputNormal->size() << " normals" << std::endl;
}

/*法向量滤波*/
void normalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
    pcl::PointCloud<pcl::Normal>::Ptr inputNormal,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud,
    pcl::PointCloud<pcl::Normal>::Ptr outputNormal)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*inputCloud, *inputNormal, *cloud_normals);

    pcl::PassThrough<pcl::PointNormal> pass;
    pass.setInputCloud(cloud_normals);
    pass.setFilterFieldName("normal_y");
    pass.setFilterLimits(-0.5, 0.5);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals_filtered(new pcl::PointCloud<pcl::PointNormal>);
    pass.filter(*cloud_normals_filtered);

    pcl::copyPointCloud(*cloud_normals_filtered, *outputCloud);
    pcl::copyPointCloud(*cloud_normals_filtered, *outputNormal);
    std::cout << "have " << outputCloud->size() << " points after Normal Filter" << std::endl;
}

/*区域生长*/
void useRegionGrow(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
    pcl::PointCloud<pcl::Normal>::Ptr inputNormal,
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree,
    std::vector <pcl::PointIndices>* clusters)
{
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    //reg.setMinClusterSize(10000); //0.005
    reg.setMinClusterSize(7000); //0.01
    reg.setMaxClusterSize(10000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(50);
    reg.setInputCloud(inputCloud);
    //reg.setIndices (indices);
    reg.setInputNormals(inputNormal);
    //reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
    //reg.setCurvatureThreshold(0.15);
    reg.setSmoothnessThreshold(3 / 180.0 * M_PI);
    reg.setCurvatureThreshold(0.8);
    reg.extract(*clusters);
}

/*计算修正主成分方向与法向量方向*/
void computePCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster,
    Eigen::Matrix3f* eigenVectorsPCA,
    Eigen::Vector4f* pcaCentroid)
{
    // 计算主成分方向
    pcl::compute3DCentroid(*cluster, *pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cluster, *pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    
    Eigen::Matrix3f EigenVectors;
    EigenVectors = eigen_solver.eigenvectors();
    //eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    
    // 竖直方向，用于辅助修正主成分方向
    Eigen::Vector3f Vvertical(0, -1, 0);

    Eigen::Vector3f Vz;
    Eigen::Vector3f Vy;
    Eigen::Vector3f Vx;

    // 法向量
    EigenVectors.col(0).normalize();
    Vz = EigenVectors.col(0);

    // 修正竖直方向
    EigenVectors.col(1) = Vvertical - (EigenVectors.col(0).dot(Vvertical) * EigenVectors.col(0));
    EigenVectors.col(1).normalize();
    Vy = EigenVectors.col(1);

    // 横向
    EigenVectors.col(2) = EigenVectors.col(0).cross(EigenVectors.col(1));
    EigenVectors.col(2).normalize();
    Vx = EigenVectors.col(2);

    eigenVectorsPCA->col(0) = Vx;
    eigenVectorsPCA->col(1) = Vy;
    eigenVectorsPCA->col(2) = Vz;
}

/*
// RANSAC提取直线
void extractLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,std::vector<pcl::PointCloud<pcl::PointXYZ>> lines) {

    // 使用RANSAC拟合直线模型
    pcl::ModelCoefficients::Ptr coefficients;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setMaxIterations(1000);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // 循环迭代，提取所有墙面
    while (true)
    {
        // 直线提取
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        std::cout << "Found " << inliers->indices.size() << " inliers" << std::endl;

        // 检查是否检测到直线
        if (inliers->indices.size() < 10) {
            break;
        }

        // 储存平面点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_line(new pcl::PointCloud<pcl::PointXYZ>);

        // 提取平面模型内的点云数据
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_line);
        // 将墙面储存
        //std::cout << "this plane has " << cloud_plane->size() << " points" << std::endl;
        lines.push_back(*cloud_line);
        std::cout << "Downsampled cloud contains " << downsampled_cloud->size() << " data points" << std::endl;
        std::cout << "already detect " << all_walls.size() << " walls" << std::endl;


        // 移除平面对应的点云
        extract.setNegative(true);
        extract.filter(*downsampled_cloud);
        // 移除对应点云对应的法向量
        pcl::ExtractIndices<pcl::Normal> extract_normals;  // 使用ExtractIndices类
        extract_normals.setInputCloud(normals);
        extract_normals.setIndices(inliers);
        extract_normals.setNegative(true);
        extract_normals.filter(*normals);  // 将平面以及对应的法向量从点云中滤除
        //break;
    }
    // 获取参数a、b、c并计算两端点
    float a = coefficients.values[0], b = coefficients.values[1], c = coefficients.values[3];
    pcl::PointXYZ pt1, pt2;
    // 这里假设平面方程为z=0,只需要计算x和y
    pt1.x = -c / a; pt1.y = -c / b; pt1.z = 0;
    pt2.x = (-c - a * cloud->points[inliers[0]].x) / b;
    pt2.y = (-c - b * cloud->points[inliers[0]].y) / a; pt2.z = 0;

    // 对内点进行投影
    for (int idx : inliers) {
        pcl::PointXYZ& pt = cloud->points[idx];
        float x = pt.x, y = pt.y;
        float x' = (b*(bx - ay) - c) / (a*a + b*b);
        float y' = (a*(ax - by) - c) / (a*a + b*b); 
        pt.x = x'; pt.y = y';
    }

    // 显示原点云、内点和两端点  
    pcl::visualization::PCLVisualizer vis;
    //...  显示点云和处理结果
}
*/

//根据索引提取所需点云，indices为索引，negative为false时提取索引所示点云，为true提取剩余点云（因为调用的是滤波器，滤掉true和false，所以提取的点逻辑相反
pcl::PointCloud<pcl::PointXYZ>::Ptr filterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                pcl::PointIndices::Ptr indices,
                                                bool negative,
                                                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
    pcl::ExtractIndices<pcl::PointXYZ> extract;//include<pcl/filters/extract_indices.h>
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.filter(*cloud_filtered);

    return cloud_filtered;
}

void Seg_RansacLine(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    pcl::PointIndices::Ptr inliers,
                    pcl::ModelCoefficients::Ptr coefficients,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outliers,
                    double distance_threshold)
{
    int max_iterations = 1000;
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);//选择提取的模型为直线（PCL中预设好的）
    seg.setMethodType(pcl::SAC_RANSAC);//分割方法选择为RANSAC
    seg.setDistanceThreshold(distance_threshold);
    seg.setMaxIterations(max_iterations);//最大迭代次数
    seg.setInputCloud(cloud);//输入
    seg.segment(*inliers, *coefficients);//分割
    
    filterCloud(cloud, inliers, false, cloud_inliers);//提取平面中的点
    filterCloud(cloud, inliers, true, cloud_outliers);//提取非平面中的点

    return;
}

//提取一条直线并存储
bool extractLineByRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left,
                  std::vector<pcl::ModelCoefficients::Ptr> *horizontal_lines,
                  std::vector<pcl::ModelCoefficients::Ptr> *vertical_lines)
{
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers(new pcl::PointCloud<pcl::PointXYZ>);
    
    std::cout << "cloud_input has " << cloud_input->size() << "points" << endl;
    Seg_RansacLine(cloud_input, inliers, coefficients, cloud_inliers, cloud_left, 0.02);
    std::cout << "cloud_left has " << cloud_left->size() << "points" << endl;

    std::cout << "coefficients: " << " 0: " << coefficients->values[0] << " 1: " << coefficients->values[1] << " 2: " << coefficients->values[2] << " 3: " << coefficients->values[3] << " 4: " << coefficients->values[4] << endl;
    if (abs(coefficients->values[3]) < 0.1)
    {
        vertical_lines->push_back(coefficients);
        std::cout << "line is vertical!" << endl;
    }
    else if (abs(coefficients->values[4]) < 0.1)
    {
        horizontal_lines->push_back(coefficients);
        std::cout << "line is horizontal!" << endl;
    }
    else
    {
        std::cout << "line is neither vertical or horizontal!" << endl;
    }

    if (cloud_left->points.size() < 10) {
        std::cout << "don't have enough points" << endl;
        return false;
    }
    return true;
}

//提取所有直线
void allLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input,
              std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines,
              std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines)
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outliers(new pcl::PointCloud<pcl::PointXYZ>);
    int max_lines_num = 20;
    if (cloud_input->size() < 5) {
        std::cout << "only have " << cloud_input->size() << " points" << " need more points!" << endl;
    }

    while (extractLineByRANSAC(cloud_input, cloud_outliers, horizontal_lines, vertical_lines) && int(horizontal_lines->size()+ vertical_lines->size()) < max_lines_num)
    {
        cloud_input = cloud_outliers;
    }
    return;
}

struct compareHor
{
    bool operator()(pcl::ModelCoefficients::Ptr a, pcl::ModelCoefficients::Ptr b)
    {
        return (-a->values[0] * a->values[4] / a->values[3] + a->values[1]) < (-b->values[0] * b->values[4] / b->values[3] + b->values[1]);
    }
};

struct compareVer
{
    bool operator()(pcl::ModelCoefficients::Ptr a, pcl::ModelCoefficients::Ptr b)
    {
        return (-a->values[1] * a->values[3] / a->values[4] + a->values[0]) < (-b->values[1] * b->values[3] / b->values[4] + b->values[0]);
    }
};

void getIntersecPoints(std::vector<pcl::PointXYZ> *IntersecPoints,
    std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines,
    std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines)
{
    std::sort(horizontal_lines->begin(), horizontal_lines->end(), compareHor());
    std::sort(vertical_lines->begin(), vertical_lines->end(), compareVer());
    int line_i = 0;
    for (pcl::ModelCoefficients::Ptr line : *horizontal_lines) {
        std::cout << "horizontal_line_" << line_i << " : " << -line->values[0] * line->values[4] / line->values[3] + line->values[1] <<endl;
        line_i++; 
    }
    line_i = 0;
    for (pcl::ModelCoefficients::Ptr line : *vertical_lines) {
        std::cout << "vertical_line_" << line_i << " : " << -line->values[1] * line->values[3] / line->values[4] + line->values[0] << endl;
        line_i++;
    }
}

/*
void getLineCoefficients(pcl::PointXYZ pt1,
    pcl::PointXYZ pt2,
    pcl::ModelCoefficients& line_coeffs)
{
    line_coeffs.values[0] = pt1.x;
    line_coeffs.values[1] = pt1.y;
    line_coeffs.values[2] = pt1.z;

    // 方向向量 = 终点 - 起点  
    Eigen::Vector3f direction = Eigen::Vector3f(pt2.x, pt2.y, pt2.z) - Eigen::Vector3f(pt1.x, pt1.y, pt1.z);

    //  normalized 方向向量     
    direction.normalize();

    // 设置方向向量  
    line_coeffs.values[3] = direction[0];
    line_coeffs.values[4] = direction[1];
    line_coeffs.values[5] = direction[2];
}
*/

/*根据两点求直线的参数*/
void getLineCoefficients(pcl::PointXYZ pt1,
    pcl::PointXYZ pt2,
    Eigen::VectorXf& line_coeffs)
{
    line_coeffs(0) = pt1.x;
    line_coeffs(1) = pt1.y;
    line_coeffs(2) = pt1.z;

    // 方向向量 = 终点 - 起点  
    Eigen::Vector3f direction = Eigen::Vector3f(pt2.x, pt2.y, pt2.z) - Eigen::Vector3f(pt1.x, pt1.y, pt1.z);

    //  normalized 方向向量     
    direction.normalize();

    // 设置方向向量  
    line_coeffs(3) = direction[0];
    line_coeffs(4) = direction[1];
    line_coeffs(5) = direction[2];
}

/*求水平线和多边形的一条边作为线段来说是否有交点（而非作为直线）*/
bool lineSegmentIntersection(const Eigen::VectorXf& height_line_coeffs,
    const Eigen::VectorXf& polygon_line_coeffs,
    Eigen::Vector4f& intersect_point,
    pcl::PointXYZ pt1,
    pcl::PointXYZ pt2)
{
    // 因为是与水平线相交，所以只需检查多边形边两点的高度与交点的关系就可以知道线段是否相交（而不是直线）
    if (!pcl::lineWithLineIntersection(height_line_coeffs, polygon_line_coeffs, intersect_point, 1e-4))
    {
        //如果直线不相交（存在平行情况）
        return FALSE;
    }

    // 此时已不存在多边形边的两点y值相同的情况（否则已经return）
    double bigger_y = pt1.y > pt2.y ? pt1.y : pt2.y;
    double smaller_y = pt1.y < pt2.y ? pt1.y : pt2.y;

    if (intersect_point[1] < smaller_y or intersect_point[1] >= bigger_y)
    {
        return FALSE;
    }

    return TRUE;
    
}

void heightWithConvexIntersection(double height,
    double min_x,
    double max_x,
    pcl::PointCloud<pcl::PointXYZ>::Ptr Polygon_points,
    std::vector<Eigen::Vector4f>& Intersect_points)
{
    // 指定高度的水平直线
    pcl::PointXYZ pt1, pt2, pt3, pt4;
    pt1.x = min_x;
    pt1.y = height;
    pt2.x = max_x;
    pt2.y = height;
    //pcl::ModelCoefficients height_line_coeffs;
    Eigen::VectorXf height_line_coeffs(6);
    getLineCoefficients(pt1, pt2, height_line_coeffs);

    // 多边形的边
    //pcl::ModelCoefficients polygon_line_coeffs;
    Eigen::VectorXf polygon_line_coeffs(6);
    Eigen::Vector4f intersect_point;
    int intersect_points_num = 0;
    
    //std::cout << "Polygon_points_size: " << Polygon_points->size() << endl;
    //for(pcl::PointXYZ tmp_point : *Polygon_points){
    //    std::cout << "tmp_point:\n " << tmp_point << endl;
    //}

    for (int i = 0;
        i < Polygon_points->size();
        i++) 
    {
        //最后一个点
        if (i == Polygon_points->size() - 1) 
        {
            pt3 = Polygon_points->points[i];
            pt4 = Polygon_points->points[0];
        }
        else 
        {
            pt3 = Polygon_points->points[i];
            pt4 = Polygon_points->points[i+1];
        }
        getLineCoefficients(pt3, pt4, polygon_line_coeffs);
        if(lineSegmentIntersection(height_line_coeffs, polygon_line_coeffs, intersect_point, pt3, pt4))
        {
            //std::cout << "pt3: " << pt3 << endl;
            //std::cout << "pt4: " << pt4 << endl;
            //std::cout << "intersect_point: " << intersect_point << endl;
            if (intersect_points_num == 2)
            {
                std::cout << "ERROR: line with convex have more than two intersect point!" << endl;
                return;
            }
            Intersect_points[intersect_points_num] = intersect_point;
            intersect_points_num++;
            //std::cout << "get intersect point!" << endl;
        }
    }
    
    // 齐次坐标，不然在变换回原坐标系时会没有位移只有旋转
    Intersect_points[0][3] = 1;
    Intersect_points[1][3] = 1;

    // 确保第一个交点在左侧，第二个交点在右侧
    Eigen::Vector4f tmp;
    if (Intersect_points[0][0] > Intersect_points[1][0])
    {
        tmp = Intersect_points[0];
        Intersect_points[0] = Intersect_points[1];
        Intersect_points[1] = tmp;
    }
    //std::cout << "Intersect_point1: " << Intersect_points[0] << endl;
    //std::cout << "Intersect_point2: " << Intersect_points[1] << endl;
    //std::cout << "get " << Intersect_points->size() << "intersect point" << endl;
}


/*接收一个二维的凸多边形，以及将他变换回三维空间的变换矩阵，返回一条二维的路径，与在三维空间的还原路径*/
void getPathPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points, 
    Eigen::Matrix4f restoreTransform, 
    std::vector<Eigen::Vector4f>& path_points, 
    std::vector<Eigen::Vector4f>& restored_path_points)
{
    // 提取凸多边形的最大边界
    pcl::PointXYZ minHullPoint, maxHullPoint;
    pcl::getMinMax3D(*hull_points, minHullPoint, maxHullPoint);
    std::cout << "minHullPoint: " << minHullPoint << "points!" << endl;
    std::cout << "maxHullPoint: " << maxHullPoint << "points!" << endl;

    // 覆盖范围
    double cover_range = 0.2;
    // 分别用于存储全部路径点、单次得到的两个交点、还原回原坐标系的两个交点
    std::vector<Eigen::Vector4f> two_points(2);
    std::vector<Eigen::Vector4f> two_restored_points(2);
    // 路径规划从左侧开始
    bool is_Left = TRUE;
    for (double height = maxHullPoint.y - cover_range / 2;
        height > minHullPoint.y;
        height = height - cover_range)
    {
        heightWithConvexIntersection(height, minHullPoint.x, maxHullPoint.x, hull_points, two_points);
        // 将点还原到原坐标系
        two_restored_points[0] = restoreTransform * two_points[0];
        two_restored_points[1] = restoreTransform * two_points[1];
        if (is_Left)
        {
            path_points.push_back(two_points[0]);
            path_points.push_back(two_points[1]);
            restored_path_points.push_back(two_restored_points[0]);
            restored_path_points.push_back(two_restored_points[1]);
            is_Left = FALSE;
        }
        else
        {
            path_points.push_back(two_points[1]);
            path_points.push_back(two_points[0]);
            restored_path_points.push_back(two_restored_points[1]);
            restored_path_points.push_back(two_restored_points[0]);
            is_Left = TRUE;
        }
    }

    /*
    std::cout << "path_points have " << path_points->size() << "points" << endl;
    if (path_points->size() != 0)
    {
        std::cout << "points on path are: " << endl;
        for (Eigen::Vector4f each_point : *path_points) {
            std::cout << each_point << endl;
        }
    }
    */
    return ;
}

/*根据投影到平面上的分割墙面点云，获得墙面的角点*/
void getCornerPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected,
    Eigen::Matrix4f restoreTransform,
    std::vector<Eigen::Vector4f>& corner_points,
    std::vector<Eigen::Vector4f>& restored_corner_points)
{
    pcl::PointXYZ BottomLeft, TopRight;
    Eigen::Vector4f BottomLeft_eigen, TopRight_eigen;
    pcl::getMinMax3D(*cloud_projected, BottomLeft, TopRight);

    PclPointXYZ2EigenVector4f(BottomLeft, BottomLeft_eigen);
    PclPointXYZ2EigenVector4f(TopRight, TopRight_eigen);
    
    
    
    // 在x方向上翻转点云，从而找到右下角(不能 Eigen::Matrix4f reversalTransform(Eigen::Matrix4f::Identity()); restoreTransform(0,0) = -1;)
    Eigen::Matrix4f reversalTransform;
    reversalTransform.setIdentity();
    reversalTransform(0, 0) = -1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reversal(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_projected, *cloud_reversal, reversalTransform);

    pcl::PointXYZ BottomRight, TopLeft;
    Eigen::Vector4f BottomRight_eigen, TopLeft_eigen;
    pcl::getMinMax3D(*cloud_projected, BottomRight, TopLeft);

    PclPointXYZ2EigenVector4f(BottomRight, BottomRight_eigen);
    PclPointXYZ2EigenVector4f(TopLeft, TopLeft_eigen);

    // 翻转回原投影坐标
    BottomRight_eigen = reversalTransform * BottomRight_eigen;
    TopLeft_eigen = reversalTransform * TopLeft_eigen;

    corner_points.push_back(TopLeft_eigen);
    corner_points.push_back(TopRight_eigen);
    corner_points.push_back(BottomLeft_eigen);
    corner_points.push_back(BottomRight_eigen);

    // 把找到的四个点都旋转回原空间
    TopLeft_eigen = restoreTransform * TopLeft_eigen;
    TopRight_eigen = restoreTransform * TopRight_eigen;
    BottomLeft_eigen = restoreTransform * BottomLeft_eigen;
    BottomRight_eigen = restoreTransform * BottomRight_eigen;

    restored_corner_points.push_back(TopLeft_eigen);
    restored_corner_points.push_back(TopRight_eigen);
    restored_corner_points.push_back(BottomLeft_eigen);
    restored_corner_points.push_back(BottomRight_eigen);
}

/*点类型转换*/
void PclPointXYZ2EigenVector4f(pcl::PointXYZ& pt_pcl,
    Eigen::Vector4f& pt_eigen)
{
    pt_eigen[0] = pt_pcl.x;
    pt_eigen[1] = pt_pcl.y;
    pt_eigen[2] = pt_pcl.z;
    pt_eigen[3] = 1;
    
}

void EigenVector4f2PclPointXYZ(Eigen::Vector4f& pt_eigen, 
    pcl::PointXYZ& pt_pcl)
{
    pt_pcl.x = pt_eigen[0];
    pt_pcl.y = pt_eigen[1];
    pt_pcl.z = pt_eigen[2];
}

// 借助主成分分析得到平面法向量方向，进而得到bounding box
/*
void getBBbyPCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, pcl::PointXYZ *min_pt, pcl::PointXYZ *max_pt, Eigen::Quaternionf *bboxQuaternion, Eigen::Vector3f *bboxTransform) {
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cluster, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cluster, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    //eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    Eigen::Vector3f Vvertical(0, -1, 0);
    std::cout << "eigenVectorPCA is: \n" << eigenVectorsPCA << endl;
    // 法向量
    eigenVectorsPCA.col(0).normalize();
    // 主方向
    eigenVectorsPCA.col(1) = Vvertical - (eigenVectorsPCA.col(0).dot(Vvertical) * eigenVectorsPCA.col(0));
    eigenVectorsPCA.col(1).normalize();
    // 横向
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    eigenVectorsPCA.col(2).normalize();
    std::cout << "eigenVectorPCA after normalize is: \n" << eigenVectorsPCA << endl;
    // Transform the original cloud to the origin where the principal components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cluster, *cloudPointsProjected, projectionTransform);

    // Get the minimum and maximum points of the transformed cloud.
    pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());
    
    
    return;
}
*/

// 用于边界点简化的结构体以及函数
struct myFlag
{
    int startflag;
    int endflag;
};

double mydistance(pcl::PointXYZ start, pcl::PointXYZ end, pcl::PointXYZ point)
{
    double x, y, z, distance;
    pcl::PointXYZ v1, v2;
    v1.x = start.x - end.x;
    v1.y = start.y - end.y;
    v1.z = start.z - end.z;
    v2.x = start.x - point.x;
    v2.y = start.y - point.y;
    v2.z = start.z - point.z;
    x = v1.y * v2.z - v1.z * v2.y;
    y = v1.z * v2.x - v1.x * v2.z;
    z = v1.x * v2.y - v1.y * v2.x;

    distance = sqrt(x * x + y * y + z * z) / sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);

    return distance;
}

void LineBoun(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_results)
{
    boundary_results->clear();
    std::vector<myFlag> myVf;
    double thre = 0.02;
    std::stack<myFlag> mySf;
    myFlag f;
    f.endflag = cloud_boundary->size() - 1;
    f.startflag = 0;
    mySf.push(f);

    while (!mySf.empty())
    {
        myFlag fi;
        pcl::PointXYZ start, end;
        fi = mySf.top();
        mySf.pop();
        start = cloud_boundary->points[fi.startflag];
        end = cloud_boundary->points[fi.endflag];
        int pos = 0;
        double maxdis = 0;
        for (int i = fi.startflag + 1; i < fi.endflag; i++)
        {
            double dis = mydistance(start, end, cloud_boundary->points[i]);
            if (dis > maxdis)
            {
                pos = i;
                maxdis = dis;
            }
        }
        if (maxdis > thre)
        {
            myFlag fnew;
            fnew.startflag = fi.startflag;
            fnew.endflag = pos;
            mySf.push(fnew);
            fnew.startflag = pos;
            fnew.endflag = fi.endflag;
            mySf.push(fnew);
        }
        else
        {
            myVf.push_back(fi);
        }
    }
    /*for (int i = 1; i < myVf.size(); i++)
    {
        cloud_newboun->points.push_back(cloud_boundary->points[myVf[i].startflag]);
    }*/
    for (int i = 0; i < myVf.size(); i++)
    {
        boundary_results->points.push_back(cloud_boundary->points[myVf[i].startflag]);
    }
    return;
}

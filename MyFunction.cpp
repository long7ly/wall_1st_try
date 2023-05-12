#include "MyFunction.h"

/*��ȡ��������*/
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

/*�ص����� ����ѡ���������*/
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

/*ͳ���˲�*/
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

/*�����˲�*/
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

/*������*/
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

/*���㷨����*/
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

/*�������˲�*/
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

/*��������*/
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

/*�����������ɷַ����뷨��������*/
void computePCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster,
    Eigen::Matrix3f* eigenVectorsPCA,
    Eigen::Vector4f* pcaCentroid)
{
    // �������ɷַ���
    pcl::compute3DCentroid(*cluster, *pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cluster, *pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    
    Eigen::Matrix3f EigenVectors;
    EigenVectors = eigen_solver.eigenvectors();
    //eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    
    // ��ֱ�������ڸ����������ɷַ���
    Eigen::Vector3f Vvertical(0, -1, 0);

    Eigen::Vector3f Vz;
    Eigen::Vector3f Vy;
    Eigen::Vector3f Vx;

    // ������
    EigenVectors.col(0).normalize();
    Vz = EigenVectors.col(0);

    // ������ֱ����
    EigenVectors.col(1) = Vvertical - (EigenVectors.col(0).dot(Vvertical) * EigenVectors.col(0));
    EigenVectors.col(1).normalize();
    Vy = EigenVectors.col(1);

    // ����
    EigenVectors.col(2) = EigenVectors.col(0).cross(EigenVectors.col(1));
    EigenVectors.col(2).normalize();
    Vx = EigenVectors.col(2);

    eigenVectorsPCA->col(0) = Vx;
    eigenVectorsPCA->col(1) = Vy;
    eigenVectorsPCA->col(2) = Vz;
}

/*
// RANSAC��ȡֱ��
void extractLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,std::vector<pcl::PointCloud<pcl::PointXYZ>> lines) {

    // ʹ��RANSAC���ֱ��ģ��
    pcl::ModelCoefficients::Ptr coefficients;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setMaxIterations(1000);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // ѭ����������ȡ����ǽ��
    while (true)
    {
        // ֱ����ȡ
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        std::cout << "Found " << inliers->indices.size() << " inliers" << std::endl;

        // ����Ƿ��⵽ֱ��
        if (inliers->indices.size() < 10) {
            break;
        }

        // ����ƽ�����
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_line(new pcl::PointCloud<pcl::PointXYZ>);

        // ��ȡƽ��ģ���ڵĵ�������
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_line);
        // ��ǽ�洢��
        //std::cout << "this plane has " << cloud_plane->size() << " points" << std::endl;
        lines.push_back(*cloud_line);
        std::cout << "Downsampled cloud contains " << downsampled_cloud->size() << " data points" << std::endl;
        std::cout << "already detect " << all_walls.size() << " walls" << std::endl;


        // �Ƴ�ƽ���Ӧ�ĵ���
        extract.setNegative(true);
        extract.filter(*downsampled_cloud);
        // �Ƴ���Ӧ���ƶ�Ӧ�ķ�����
        pcl::ExtractIndices<pcl::Normal> extract_normals;  // ʹ��ExtractIndices��
        extract_normals.setInputCloud(normals);
        extract_normals.setIndices(inliers);
        extract_normals.setNegative(true);
        extract_normals.filter(*normals);  // ��ƽ���Լ���Ӧ�ķ������ӵ������˳�
        //break;
    }
    // ��ȡ����a��b��c���������˵�
    float a = coefficients.values[0], b = coefficients.values[1], c = coefficients.values[3];
    pcl::PointXYZ pt1, pt2;
    // �������ƽ�淽��Ϊz=0,ֻ��Ҫ����x��y
    pt1.x = -c / a; pt1.y = -c / b; pt1.z = 0;
    pt2.x = (-c - a * cloud->points[inliers[0]].x) / b;
    pt2.y = (-c - b * cloud->points[inliers[0]].y) / a; pt2.z = 0;

    // ���ڵ����ͶӰ
    for (int idx : inliers) {
        pcl::PointXYZ& pt = cloud->points[idx];
        float x = pt.x, y = pt.y;
        float x' = (b*(bx - ay) - c) / (a*a + b*b);
        float y' = (a*(ax - by) - c) / (a*a + b*b); 
        pt.x = x'; pt.y = y';
    }

    // ��ʾԭ���ơ��ڵ�����˵�  
    pcl::visualization::PCLVisualizer vis;
    //...  ��ʾ���ƺʹ�����
}
*/

//����������ȡ������ƣ�indicesΪ������negativeΪfalseʱ��ȡ������ʾ���ƣ�Ϊtrue��ȡʣ����ƣ���Ϊ���õ����˲������˵�true��false��������ȡ�ĵ��߼��෴
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
    seg.setModelType(pcl::SACMODEL_LINE);//ѡ����ȡ��ģ��Ϊֱ�ߣ�PCL��Ԥ��õģ�
    seg.setMethodType(pcl::SAC_RANSAC);//�ָ��ѡ��ΪRANSAC
    seg.setDistanceThreshold(distance_threshold);
    seg.setMaxIterations(max_iterations);//����������
    seg.setInputCloud(cloud);//����
    seg.segment(*inliers, *coefficients);//�ָ�
    
    filterCloud(cloud, inliers, false, cloud_inliers);//��ȡƽ���еĵ�
    filterCloud(cloud, inliers, true, cloud_outliers);//��ȡ��ƽ���еĵ�

    return;
}

//��ȡһ��ֱ�߲��洢
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

//��ȡ����ֱ��
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

    // �������� = �յ� - ���  
    Eigen::Vector3f direction = Eigen::Vector3f(pt2.x, pt2.y, pt2.z) - Eigen::Vector3f(pt1.x, pt1.y, pt1.z);

    //  normalized ��������     
    direction.normalize();

    // ���÷�������  
    line_coeffs.values[3] = direction[0];
    line_coeffs.values[4] = direction[1];
    line_coeffs.values[5] = direction[2];
}
*/

/*����������ֱ�ߵĲ���*/
void getLineCoefficients(pcl::PointXYZ pt1,
    pcl::PointXYZ pt2,
    Eigen::VectorXf& line_coeffs)
{
    line_coeffs(0) = pt1.x;
    line_coeffs(1) = pt1.y;
    line_coeffs(2) = pt1.z;

    // �������� = �յ� - ���  
    Eigen::Vector3f direction = Eigen::Vector3f(pt2.x, pt2.y, pt2.z) - Eigen::Vector3f(pt1.x, pt1.y, pt1.z);

    //  normalized ��������     
    direction.normalize();

    // ���÷�������  
    line_coeffs(3) = direction[0];
    line_coeffs(4) = direction[1];
    line_coeffs(5) = direction[2];
}

/*��ˮƽ�ߺͶ���ε�һ������Ϊ�߶���˵�Ƿ��н��㣨������Ϊֱ�ߣ�*/
bool lineSegmentIntersection(const Eigen::VectorXf& height_line_coeffs,
    const Eigen::VectorXf& polygon_line_coeffs,
    Eigen::Vector4f& intersect_point,
    pcl::PointXYZ pt1,
    pcl::PointXYZ pt2)
{
    // ��Ϊ����ˮƽ���ཻ������ֻ�������α�����ĸ߶��뽻��Ĺ�ϵ�Ϳ���֪���߶��Ƿ��ཻ��������ֱ�ߣ�
    if (!pcl::lineWithLineIntersection(height_line_coeffs, polygon_line_coeffs, intersect_point, 1e-4))
    {
        //���ֱ�߲��ཻ������ƽ�������
        return FALSE;
    }

    // ��ʱ�Ѳ����ڶ���αߵ�����yֵ��ͬ������������Ѿ�return��
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
    // ָ���߶ȵ�ˮƽֱ��
    pcl::PointXYZ pt1, pt2, pt3, pt4;
    pt1.x = min_x;
    pt1.y = height;
    pt2.x = max_x;
    pt2.y = height;
    //pcl::ModelCoefficients height_line_coeffs;
    Eigen::VectorXf height_line_coeffs(6);
    getLineCoefficients(pt1, pt2, height_line_coeffs);

    // ����εı�
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
        //���һ����
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
    
    // ������꣬��Ȼ�ڱ任��ԭ����ϵʱ��û��λ��ֻ����ת
    Intersect_points[0][3] = 1;
    Intersect_points[1][3] = 1;

    // ȷ����һ����������࣬�ڶ����������Ҳ�
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


/*����һ����ά��͹����Σ��Լ������任����ά�ռ�ı任���󣬷���һ����ά��·����������ά�ռ�Ļ�ԭ·��*/
void getPathPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points, 
    Eigen::Matrix4f restoreTransform, 
    std::vector<Eigen::Vector4f>& path_points, 
    std::vector<Eigen::Vector4f>& restored_path_points)
{
    // ��ȡ͹����ε����߽�
    pcl::PointXYZ minHullPoint, maxHullPoint;
    pcl::getMinMax3D(*hull_points, minHullPoint, maxHullPoint);
    std::cout << "minHullPoint: " << minHullPoint << "points!" << endl;
    std::cout << "maxHullPoint: " << maxHullPoint << "points!" << endl;

    // ���Ƿ�Χ
    double cover_range = 0.2;
    // �ֱ����ڴ洢ȫ��·���㡢���εõ����������㡢��ԭ��ԭ����ϵ����������
    std::vector<Eigen::Vector4f> two_points(2);
    std::vector<Eigen::Vector4f> two_restored_points(2);
    // ·���滮����࿪ʼ
    bool is_Left = TRUE;
    for (double height = maxHullPoint.y - cover_range / 2;
        height > minHullPoint.y;
        height = height - cover_range)
    {
        heightWithConvexIntersection(height, minHullPoint.x, maxHullPoint.x, hull_points, two_points);
        // ���㻹ԭ��ԭ����ϵ
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

/*����ͶӰ��ƽ���ϵķָ�ǽ����ƣ����ǽ��Ľǵ�*/
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
    
    
    
    // ��x�����Ϸ�ת���ƣ��Ӷ��ҵ����½�(���� Eigen::Matrix4f reversalTransform(Eigen::Matrix4f::Identity()); restoreTransform(0,0) = -1;)
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

    // ��ת��ԭͶӰ����
    BottomRight_eigen = reversalTransform * BottomRight_eigen;
    TopLeft_eigen = reversalTransform * TopLeft_eigen;

    corner_points.push_back(TopLeft_eigen);
    corner_points.push_back(TopRight_eigen);
    corner_points.push_back(BottomLeft_eigen);
    corner_points.push_back(BottomRight_eigen);

    // ���ҵ����ĸ��㶼��ת��ԭ�ռ�
    TopLeft_eigen = restoreTransform * TopLeft_eigen;
    TopRight_eigen = restoreTransform * TopRight_eigen;
    BottomLeft_eigen = restoreTransform * BottomLeft_eigen;
    BottomRight_eigen = restoreTransform * BottomRight_eigen;

    restored_corner_points.push_back(TopLeft_eigen);
    restored_corner_points.push_back(TopRight_eigen);
    restored_corner_points.push_back(BottomLeft_eigen);
    restored_corner_points.push_back(BottomRight_eigen);
}

/*������ת��*/
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

// �������ɷַ����õ�ƽ�淨�������򣬽����õ�bounding box
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
    // ������
    eigenVectorsPCA.col(0).normalize();
    // ������
    eigenVectorsPCA.col(1) = Vvertical - (eigenVectorsPCA.col(0).dot(Vvertical) * eigenVectorsPCA.col(0));
    eigenVectorsPCA.col(1).normalize();
    // ����
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

// ���ڱ߽��򻯵Ľṹ���Լ�����
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

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/octree/octree.h>
#include <vtkRenderWindow.h>


// 自用函数
#include "MyFunction.h"

/*
void showPointCloud(pcl::visualization::PCLVisualizer Viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string cloud_name) {
    Viewer.addPointCloud<pcl::PointXYZ>(cloud, cloud_name);
    Viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, cloud_name);
    Viewer.addCoordinateSystem(0.5);
    Viewer.spin();
}
*/

std::string ReadPointCloudFileName = "./point_cloud_data/useful/single_wall_Gemini_7_Layer_addtop_down0_005.pcd";

std::string SaveFileNameIndex = "../test_MFC/point_cloud_data/cloud2/";

std::string SaveFileName_Raw = SaveFileNameIndex + "cloud_raw.pcd";
std::string SaveFileName_Filtered = SaveFileNameIndex + "cloud_filtered.pcd";
std::string SaveFileName_Segment = SaveFileNameIndex + "cluster_";
std::string SaveFileName_Edge = SaveFileNameIndex + "edge_";
std::string SaveFileName_Info = SaveFileNameIndex + "cluster_info.txt";

//std::string SaveFileName_Info = SaveFileNameIndex + "path_points_info_";

int main(int argc, char* argv[]) {
    // 初始化Qt
    //QApplication app(argc, argv);

    //过程点云可视化
    pcl::visualization::PCLVisualizer viewer("process_cloud");
    viewer.setBackgroundColor(0, 0, 0);

    // 读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    readPointCloud(ReadPointCloudFileName, cloud);
    
    // 用于储存点云，为界面显示提供数据
    pcl::PCDWriter writer;
    // 储存原点云格式
    writer.write<pcl::PointXYZ>(SaveFileName_Raw, *cloud);

    // 下采样
    //pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    downSample(cloud, cloud);
    //viewer.addPointCloud(cloud, "cloud");
    //viewer.addCoordinateSystem(0.5);
    //viewer.spin();
    //viewer.removeAllPointClouds();

    // 统计滤波
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    //statisticalFilter(cloud, cloud);
    //viewer.addPointCloud(cloud_filtered, "cloud_filtered");
    //viewer.addCoordinateSystem(0.5);
    //viewer.spin();
    //viewer.removeAllPointClouds();
    
    // 距离滤波
    distanceFilter(cloud, cloud);
    //viewer.addPointCloud(cloud, "cloud");
    //viewer.addCoordinateSystem(0.5);
    //viewer.spin();
    //viewer.removeAllPointClouds();

    //pcl::io::savePCDFile("pcd_filtered.pcd", *cloud_filtered);
    //std::cout << "Saved " << cloud_filtered->size() << " data points to test_pcd.pcd." << std::endl;
    // 移除已显示点云
    //viewer.removeAllPointClouds();

    //pcl::visualization::PCLVisualizer downsampled_cloud_viewer("downsampled_cloud");
    //showPointCloud(downsampled_cloud_viewer, downsampled_cloud, "downsampled_cloud");

    // 计算KdTree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    
    // 计算法向量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    computeNormal(cloud, tree, normals);
    //pcl::visualization::PCLVisualizer normalViewer("normal");
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::Normal> color_handler_normals(normals, 0, 0, 255);
    //normalViewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(downsampled_cloud, normals, 10, 0.05, "normals");

    // 使用法向量滤波，滤除掉地面
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Normfiltered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_Normfiltered(new pcl::PointCloud<pcl::Normal>);
    normalFilter(cloud, normals, cloud_Normfiltered, normals_Normfiltered);
    //viewer.addPointCloud(cloud, "cloud");
    //viewer.addCoordinateSystem(0.5);
    //viewer.spin();
    //viewer.removeAllPointClouds();
    
    // 储存预处理点云
    writer.write<pcl::PointXYZ>(SaveFileName_Filtered, *cloud_Normfiltered);
    
    // 区域分割
    std::vector <pcl::PointIndices> * clusters(new std::vector <pcl::PointIndices>);
    useRegionGrow(cloud_Normfiltered, normals_Normfiltered, tree, clusters);
    std::cout << "Found " << clusters->size() << " clusters" << std::endl;
    
    std::ofstream file(SaveFileName_Info);
    file << clusters->size();
    file.close();

    //原点云可视
    pcl::visualization::PCLVisualizer Rawviewer("Raw Cloud");
    Rawviewer.setBackgroundColor(0, 0, 0);
    Rawviewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");
    Rawviewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "cloud");
    //Rawviewer.addCoordinateSystem(0.5);
    /*对每块点云处理*/
    //分割点云可视化
    pcl::visualization::PCLVisualizer segment_viewer("Segment Cloud");
    segment_viewer.setBackgroundColor(0, 0, 0);
    pcl::visualization::PCLVisualizer concave_viewer("Segment polygon");
    segment_viewer.setBackgroundColor(0, 0, 0);
    
    // 对每个区域进行处理
    for (size_t i = 0; i < clusters->size(); ++i) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*cloud_Normfiltered, (*clusters)[i], *cluster);

        // 储存分割点云
        writer.write<pcl::PointXYZ>(SaveFileName_Segment + std::to_string(i) + ".pcd", *cluster);

        std::stringstream ss;
        ss << "cluster_" << i;
        std::cout << ss.str() << " has " << cluster->size() << " points" << std::endl;
        std::stringstream bb;
        bb << "bounding_box_" << i;
        
        Eigen::Matrix3f* eigenVectorsPCA(new Eigen::Matrix3f);
        Eigen::Vector4f* pcaCentroid(new Eigen::Vector4f);
        computePCA(cluster, eigenVectorsPCA, pcaCentroid);
        
        // 根据法向量与修正后的主成分方向进行坐标变换
        // 记录变换的变换矩阵
        Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
        projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA->transpose(); //逆
        projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid->head<3>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cluster, *cloud_transformed, projectionTransform);
        // 记录还原的变换矩阵
        Eigen::Matrix4f restoreTransform(Eigen::Matrix4f::Identity());
        restoreTransform.block<3, 3>(0, 0) = eigenVectorsPCA->matrix();
        restoreTransform.block<3, 1>(0, 3) = 1.f * pcaCentroid->head<3>();

        std::cout << "show cloud_transformed" << endl;
        viewer.addPointCloud(cloud_transformed, ss.str());
        viewer.addCoordinateSystem(0.5);
        viewer.spin();
        viewer.removeAllPointClouds();

        // 设定投影平面
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        coefficients->values.resize(4);
        coefficients->values[0] = coefficients->values[1] = 0;
        coefficients->values[2] = 1;
        coefficients->values[3] = 0;

        // 投影
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(cloud_transformed);
        proj.setModelCoefficients(coefficients);
        proj.filter(*cloud_projected);
        
        //std::cout << "show cloud projected" << endl;
        //viewer.addPointCloud(cloud_projected, ss.str());
        //viewer.addCoordinateSystem(0.5);
        //viewer.spin();
        //viewer.removeAllPointClouds();

        /*计算角点作为特征点*/
        std::vector<Eigen::Vector4f> corner_points;
        std::vector<Eigen::Vector4f> restored_corner_points;
        getCornerPoints(cloud_projected, restoreTransform, corner_points, restored_corner_points);

        pcl::PointCloud<pcl::PointXYZ>::Ptr corner_cloud(new pcl::PointCloud<pcl::PointXYZ>), restored_corner_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointXYZ corner_pt, restored_corner_pt;
        for (int i = 0;
            i < corner_points.size();
            i++)
        {
            EigenVector4f2PclPointXYZ(corner_points[i], corner_pt);
            corner_cloud->points.push_back(corner_pt);

            EigenVector4f2PclPointXYZ(restored_corner_points[i], restored_corner_pt);
            restored_corner_cloud->points.push_back(restored_corner_pt);
        }

        // 计算凹多边形
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ConcaveHull<pcl::PointXYZ> chull;
        //pcl::ConvexHull<pcl::PointXYZ> chull;
        chull.setInputCloud(cloud_projected);
        chull.setAlpha(0.5);
        chull.reconstruct(*cloud_hull);
        
        /*
        // 计算直线分区
        std::vector<pcl::ModelCoefficients::Ptr>* horizontal_lines(new std::vector<pcl::ModelCoefficients::Ptr>);
        std::vector<pcl::ModelCoefficients::Ptr>* vertical_lines(new std::vector<pcl::ModelCoefficients::Ptr>);

        allLines(cloud_hull, horizontal_lines, vertical_lines);
        std::vector<pcl::PointXYZ>* IntersecPoints(new std::vector<pcl::PointXYZ>);
        getIntersecPoints(IntersecPoints, horizontal_lines, vertical_lines);
        

        // 可视化线
        viewer.addPointCloud(cloud_projected, ss.str());
        int line_i = 0;
        for(pcl::ModelCoefficients::Ptr line : *horizontal_lines){
            std::stringstream horizontal_line_name;
            horizontal_line_name << "horizontal_line_" << line_i;
            line_i++;
            viewer.addLine(*line, horizontal_line_name.str());
        }
        line_i = 0;
        for (pcl::ModelCoefficients::Ptr line : *vertical_lines) {
            std::stringstream vertical_line_name;
            vertical_line_name << "vertical_line_" << line_i;
            line_i++;
            viewer.addLine(*line, vertical_line_name.str());
        }
        viewer.addCoordinateSystem(0.5);
        viewer.spin();
        viewer.removeAllPointClouds();
        */

        std::cout << "show cloud hull" << endl;
        viewer.addPointCloud(cloud_hull, ss.str());
        viewer.addCoordinateSystem(0.2);
        viewer.registerPointPickingCallback(pp_callback, (void*)&viewer);
        viewer.spin();
        viewer.removeAllPointClouds();

        // 简化边界点+边界线可视化
        pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_points(new pcl::PointCloud<pcl::PointXYZ>);
        LineBoun(cloud_hull, boundary_points);
        viewer.addPointCloud(boundary_points);
        viewer.spin();
        viewer.removeAllPointClouds();
        viewer.addPolygon<pcl::PointXYZ>(boundary_points,1,0,0);
        //viewer.addPointCloud(cloud_projected);
        viewer.spin();
        viewer.removeAllShapes();
        viewer.removeAllPointClouds();
  
        // 提取凸包
        pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ConvexHull<pcl::PointXYZ> hull3d;
        hull3d.setInputCloud(cloud_hull);
        hull3d.reconstruct(*hull_points);
        std::cout << "hull_points have "<< hull_points->size()<< "points!" << endl;
        // 凸包的最大边界
        pcl::PointXYZ minHullPoint, maxHullPoint;
        pcl::getMinMax3D(*hull_points, minHullPoint, maxHullPoint);
        std::cout << "minHullPoint: " << minHullPoint << "points!" << endl;
        std::cout << "maxHullPoint: " << maxHullPoint << "points!" << endl;
        
        std::cout << "show convex" << endl;
        viewer.addPolygon<pcl::PointXYZ>(hull_points);
        
        //viewer.addPointCloud(corner_cloud, "point");


        //viewer.addCoordinateSystem(0.5);
        viewer.registerPointPickingCallback(pp_callback, (void*)&viewer);
        

        /*绘制边界*/
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_restored(new pcl ::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud_hull, *cloud_restored, restoreTransform);
        //pcl::transformPointCloud(*cloud_hull, *cloud_restored, restoreTransform);
        // pcl::transformPointCloud(*cloud_hull, *cloud_restored, bboxTransform, bboxQuaternion);
        //concave_viewer.addPointCloud(cluster, ss.str());
        //concave_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, ss.str());
        //concave_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, ss.str());
        //concave_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, (double)i / clusters->size(), 0.0, 1.0 - (double)i / clusters->size(), ss.str());
        concave_viewer.addPolygon<pcl::PointXYZ>(cloud_restored, bb.str());
        //concave_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, ss.str());
        concave_viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, bb.str());
        
        //viewer.addPointCloud(corner_cloud, "point");
        
        
        // 储存分割边界
        writer.write<pcl::PointXYZ>(SaveFileName_Edge + std::to_string(i) + ".pcd", *cloud_restored);
        
        /*计算末端路径点*/
        std::vector<Eigen::Vector4f> path_points;
        std::vector<Eigen::Vector4f> restored_path_points;
        getPathPoints(hull_points, restoreTransform, path_points, restored_path_points);

        /*绘制路径*/
        pcl::PointXYZ pt1;
        pcl::PointXYZ pt2;
        for (int i = 0;
            i < path_points.size()-1;
            i++)
        {
            EigenVector4f2PclPointXYZ(path_points[i], pt1);
            EigenVector4f2PclPointXYZ(path_points[i+1], pt2);  
            viewer.addLine(pt1, pt2, 1, 0, 0, ss.str()+"line"+ std::to_string(i));

            EigenVector4f2PclPointXYZ(restored_path_points[i], pt1);
            EigenVector4f2PclPointXYZ(restored_path_points[i + 1], pt2);
            concave_viewer.addLine(pt1, pt2, 1, 0, 0, ss.str() + "line" + std::to_string(i));
        }

        viewer.spin();
        viewer.removeAllPointClouds();
        viewer.removeAllShapes();


        // 绘制点云
        segment_viewer.addPointCloud(cluster, ss.str());
        segment_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, ss.str());
        segment_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, ss.str());
        segment_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, (double)i / clusters->size(), 0.0, 1.0 - (double)i / clusters->size(), ss.str());
        // 绘制关键点
        //segment_viewer.addPointCloud(restored_corner_cloud, ss.str()+"corner");
        //segment_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, ss.str() + "corner");
        //segment_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7.0, ss.str() + "corner");
        /*
        // 计算BB
        pcl::PointXYZ minPoint, maxPoint;
        pcl::getMinMax3D(*cloud_transformed, minPoint, maxPoint);

        // 计算BB还原参数并绘制BB
        const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());
        const Eigen::Quaternionf bboxQuaternion(*eigenVectorsPCA); //Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
        const Eigen::Vector3f bboxTransform = *eigenVectorsPCA * meanDiagonal + pcaCentroid->head<3>();
        segment_viewer.addCube(bboxTransform, bboxQuaternion, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z, bb.str());
        segment_viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, bb.str());
        */
    }
    //segment_viewer.addCoordinateSystem(0.5);
    std::cout << "display " << clusters->size() << " clusters" << std::endl;
    concave_viewer.registerPointPickingCallback(pp_callback, (void*)&viewer);
    segment_viewer.registerPointPickingCallback(pp_callback, (void*)&viewer);
    segment_viewer.spin();
    return 0;
}
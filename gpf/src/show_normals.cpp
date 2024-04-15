#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_types.h"
#include "pcl/ModelCoefficients.h"
#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/features/normal_3d.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/visualization/cloud_viewer.h"

#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <mutex>

class CylinderEstimationNode : public rclcpp::Node
{
public:
    CylinderEstimationNode() : Node("cylinder_estimation_node")
    {
        // Initialize subscriber to point cloud topic
        pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zed2i/left/logs/log_pointcloud", 10,
            std::bind(&CylinderEstimationNode::pointcloudCallback, this, std::placeholders::_1));

        // Create PCL visualizer
        viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer");
        viewer_->setBackgroundColor(0, 0, 0);
    }

private:
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS point cloud message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *pcl_cloud);
        // print cloud size
        std::cout << "Cloud size: " << pcl_cloud->size() << std::endl;

        pcl::visualization::CloudViewer viewer("Cloud Viewer");
        viewer.showCloud(pcl_cloud);
        while(!viewer.wasStopped())
        {
        }
        
        

        // // Compute normals
        // pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        // pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
        // pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        // ne.setSearchMethod(tree);
        // ne.setInputCloud(pcl_cloud);
        // ne.setKSearch(50); // Use 50 nearest neighbors
        // ne.compute(*normals);

        // // Lock the mutex while accessing the viewer
        // std::lock_guard<std::mutex> lock(mutex_);

        // // Remove previous point cloud and normals from viewer
        // viewer_->removeAllPointClouds();
        // viewer_->removeAllShapes();

        // // Add point cloud to viewer
        // viewer_->addPointCloud<pcl::PointXYZRGB>(pcl_cloud, "cloud");
        // viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

        // // Add normals to viewer
        // viewer_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(pcl_cloud, normals, 10, 0.05, "normals");

        // // Spin the viewer
        // viewer_->spinOnce(100);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    // std::mutex mutex_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CylinderEstimationNode>());
    rclcpp::shutdown();
    return 0;
}

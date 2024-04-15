#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_types.h"
#include "pcl/ModelCoefficients.h"
#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/filters/extract_indices.h"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

class CylinderEstimationNode : public rclcpp::Node
{
public:
    CylinderEstimationNode() : Node("cylinder_estimation_node")
    {
        // Initialize subscriber to point cloud topic
        pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zed2i/left/logs/log_pointcloud", 10,
            std::bind(&CylinderEstimationNode::pointcloudCallback, this, std::placeholders::_1));

        // Initialize publisher for marker array
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/cylinder_markers", 10);

        // Declare parameters with default values
        this->declare_parameter<double>("normal_distance_weight", 0.1);
        this->declare_parameter<int>("max_iterations", 10000);
        this->declare_parameter<double>("distance_threshold", 0.05);
        this->declare_parameter<double>("min_radius", 0.0);
        this->declare_parameter<double>("max_radius", 0.1);
    }

private:
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Fetch parameters
        double normal_distance_weight = this->get_parameter("normal_distance_weight").as_double();
        int max_iterations = this->get_parameter("max_iterations").as_int();
        double distance_threshold = this->get_parameter("distance_threshold").as_double();
        double min_radius = this->get_parameter("min_radius").as_double();
        double max_radius = this->get_parameter("max_radius").as_double();

        // Convert ROS point cloud message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *pcl_cloud);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        *remaining_cloud = *pcl_cloud;

        // Initialize marker array
        visualization_msgs::msg::MarkerArray markers;

        // Repeat RANSAC until no more cylinders are found
        while (remaining_cloud->size() > min_points_for_cylinder_)
        {
            // Perform RANSAC for cylinder estimation
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZRGB> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CYLINDER);
            seg.setMethodType(pcl::SAC_RANSAC);
            // seg.setNormalDistanceWeight(normal_distance_weight);
            seg.setMaxIterations(max_iterations);
            seg.setDistanceThreshold(distance_threshold);
            seg.setRadiusLimits(min_radius, max_radius);
            seg.setInputCloud(remaining_cloud);
            seg.segment(*inliers, *coefficients);

            // Output the cylinder coefficients
            if (inliers->indices.size() == 0)
            {
                RCLCPP_INFO(this->get_logger(), "Could not estimate a cylinder model for the remaining dataset.");
                break;
            }

            // Create a marker for the detected cylinder
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = msg->header.frame_id;
            marker.header.stamp = msg->header.stamp;
            marker.type = visualization_msgs::msg::Marker::CYLINDER;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = coefficients->values[0];
            marker.pose.position.y = coefficients->values[1];
            marker.pose.position.z = coefficients->values[2];
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = coefficients->values[6] * 2; // cylinder radius
            marker.scale.y = coefficients->values[6] * 2; // cylinder radius
            marker.scale.z = coefficients->values[5]; // cylinder height
            marker.color.a = 1.0;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            markers.markers.push_back(marker);

            // Extract the inliers of the cylinder and remove them from the remaining cloud
            pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            extract.setInputCloud(remaining_cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*remaining_cloud);
        }

        // Publish marker array
        marker_publisher_->publish(markers);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    size_t min_points_for_cylinder_ = 100; // Minimum number of points required to detect a cylinder
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CylinderEstimationNode>());
    rclcpp::shutdown();
    return 0;
}

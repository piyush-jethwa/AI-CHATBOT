<?php
$host = 'localhost'; // Usually localhost
$db   = 'medcare';
$user = 'root';      // Change if needed
$pass = '';          // Add your MySQL password if you have one

$conn = new mysqli($host, $user, $pass, $db);

if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}
?>

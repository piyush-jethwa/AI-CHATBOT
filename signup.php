<?php
require 'db.php';

$name = $_POST['signupName'];
$email = $_POST['signupEmail'];
$password = password_hash($_POST['signupPassword'], PASSWORD_BCRYPT);

// Check for existing email
$check = $conn->prepare("SELECT * FROM users WHERE email = ?");
$check->bind_param("s", $email);
$check->execute();
$result = $check->get_result();

if ($result->num_rows > 0) {
  echo "Email already registered.";
} else {
  $stmt = $conn->prepare("INSERT INTO users (name, email, password) VALUES (?, ?, ?)");
  $stmt->bind_param("sss", $name, $email, $password);
  $stmt->execute();
  echo "Registration successful!";
}

$conn->close();
?>

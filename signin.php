<?php
require 'db.php';

$email = $_POST['signinEmail'];
$password = $_POST['signinPassword'];

$stmt = $conn->prepare("SELECT * FROM users WHERE email = ?");
$stmt->bind_param("s", $email);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows === 1) {
  $user = $result->fetch_assoc();
  if (password_verify($password, $user['password'])) {
    echo "Welcome, " . $user['name'] . "!";
  } else {
    echo "Incorrect password.";
  }
} else {
  echo "User not found.";
}

$conn->close();
?>

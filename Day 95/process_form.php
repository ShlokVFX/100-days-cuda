<?php
/**
 * Student Management System - Form Processing Script
 * This script handles form submissions, database connectivity, and CRUD operations
 */

// Database connection settings
$host = "localhost";
$username = "root";
$password = "";
$database = "student_management";

// Connect to MySQL server
$conn = new mysqli($host, $username, $password);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Create database if it doesn't exist
createDatabase($conn, $database);

// Select the database
$conn->select_db($database);

// Create tables if they don't exist
createTables($conn);

// Process the request based on the action parameter
$action = isset($_REQUEST['action']) ? $_REQUEST['action'] : '';

switch ($action) {
    case 'create':
        addStudent($conn);
        break;
    case 'view':
        viewAllStudents($conn);
        break;
    case 'search':
        searchStudent($conn);
        break;
    case 'update':
        if ($_SERVER['REQUEST_METHOD'] === 'POST') {
            updateStudent($conn);
        } else {
            fetchStudentForUpdate($conn);
        }
        break;
    case 'delete':
        deleteStudent($conn);
        break;
    default:
        // Return to the form if no valid action
        header("Location: student_form.html");
        exit;
}

/**
 * Create the database if it doesn't exist
 */
function createDatabase($conn, $database) {
    $sql = "CREATE DATABASE IF NOT EXISTS $database";
    if ($conn->query($sql) !== TRUE) {
        echo "Error creating database: " . $conn->error;
    }
}

/**
 * Create tables if they don't exist
 */
function createTables($conn) {
    // Create Department table
    $sql = "CREATE TABLE IF NOT EXISTS departments (
        department_id INT AUTO_INCREMENT PRIMARY KEY,
        department_name VARCHAR(100) NOT NULL UNIQUE
    )";
    
    if ($conn->query($sql) !== TRUE) {
        echo "Error creating departments table: " . $conn->error;
    }
    
    // Create Student table
    $sql = "CREATE TABLE IF NOT EXISTS students (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        student_id VARCHAR(20) NOT NULL UNIQUE,
        dob DATE NOT NULL,
        gender VARCHAR(10) NOT NULL,
        address TEXT NOT NULL,
        mobile VARCHAR(15) NOT NULL,
        email VARCHAR(100) NOT NULL UNIQUE
    )";
    
    if ($conn->query($sql) !== TRUE) {
        echo "Error creating students table: " . $conn->error;
    }
    
    // Create the mapping table with foreign keys
    $sql = "CREATE TABLE IF NOT EXISTS student_department_mapping (
        id INT AUTO_INCREMENT PRIMARY KEY,
        student_id VARCHAR(20) NOT NULL,
        department_id INT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
        FOREIGN KEY (department_id) REFERENCES departments(department_id) ON DELETE CASCADE
    )";
    
    if ($conn->query($sql) !== TRUE) {
        echo "Error creating mapping table: " . $conn->error;
    }
}

/**
 * Add a new student to the database
 */
function addStudent($conn) {
    // Sanitize and validate input data
    $name = $conn->real_escape_string($_POST['name']);
    $student_id = $conn->real_escape_string($_POST['student_id']);
    $dob = $conn->real_escape_string($_POST['dob']);
    $gender = $conn->real_escape_string($_POST['gender']);
    $address = $conn->real_escape_string($_POST['address']);
    $mobile = $conn->real_escape_string($_POST['mobile']);
    $email = $conn->real_escape_string($_POST['email']);
    $department = $conn->real_escape_string($_POST['department']);
    
    // Start transaction
    $conn->begin_transaction();
    
    try {
        // Check if department exists, if not, create it
        $stmt = $conn->prepare("SELECT department_id FROM departments WHERE department_name = ?");
        $stmt->bind_param("s", $department);
        $stmt->execute();
        $result = $stmt->get_result();
        
        if ($result->num_rows > 0) {
            $row = $result->fetch_assoc();
            $department_id = $row['department_id'];
        } else {
            $stmt = $conn->prepare("INSERT INTO departments (department_name) VALUES (?)");
            $stmt->bind_param("s", $department);
            $stmt->execute();
            $department_id = $conn->insert_id;
        }
        
        // Insert student data
        $stmt = $conn->prepare("INSERT INTO students (name, student_id, dob, gender, address, mobile, email) VALUES (?, ?, ?, ?, ?, ?, ?)");
        $stmt->bind_param("sssssss", $name, $student_id, $dob, $gender, $address, $mobile, $email);
        
        if (!$stmt->execute()) {
            throw new Exception("Error inserting student: " . $stmt->error);
        }
        
        // Create mapping between student and department
        $stmt = $conn->prepare("INSERT INTO student_department_mapping (student_id, department_id) VALUES (?, ?)");
        $stmt->bind_param("si", $student_id, $department_id);
        
        if (!$stmt->execute()) {
            throw new Exception("Error creating mapping: " . $stmt->error);
        }
        
        // Commit transaction
        $conn->commit();
        
        displaySuccessMessage("Student added successfully!");
        
    } catch (Exception $e) {
        // Rollback transaction on error
        $conn->rollback();
        displayErrorMessage("Error: " . $e->getMessage());
    }
}

/**
 * View all students with department information
 */
function viewAllStudents($conn) {
    $sql = "SELECT s.*, d.department_name FROM students s 
            LEFT JOIN student_department_mapping m ON s.student_id = m.student_id 
            LEFT JOIN departments d ON m.department_id = d.department_id 
            ORDER BY s.name";
    
    $result = $conn->query($sql);
    
    displayHeader("All Student Records");
    
    if ($result->num_rows > 0) {
        echo "<div class='table-container'>";
        echo "<table>";
        echo "<tr>
                <th>Student ID</th>
                <th>Name</th>
                <th>Date of Birth</th>
                <th>Gender</th>
                <th>Address</th>
                <th>Mobile</th>
                <th>Email</th>
                <th>Department</th>
                <th>Actions</th>
              </tr>";
        
        while ($row = $result->fetch_assoc()) {
            echo "<tr>";
            echo "<td>" . htmlspecialchars($row['student_id']) . "</td>";
            echo "<td>" . htmlspecialchars($row['name']) . "</td>";
            echo "<td>" . htmlspecialchars($row['dob']) . "</td>";
            echo "<td>" . htmlspecialchars($row['gender']) . "</td>";
            echo "<td>" . htmlspecialchars($row['address']) . "</td>";
            echo "<td>" . htmlspecialchars($row['mobile']) . "</td>";
            echo "<td>" . htmlspecialchars($row['email']) . "</td>";
            echo "<td>" . htmlspecialchars($row['department_name']) . "</td>";
            echo "<td>
                    <a href='process_form.php?action=update&id=" . $row['student_id'] . "' class='btn edit'>Edit</a>
                    <a href='process_form.php?action=delete&id=" . $row['student_id'] . "' class='btn delete' onclick='return confirm(\"Are you sure you want to delete this record?\")'>Delete</a>
                  </td>";
            echo "</tr>";
        }
        
        echo "</table>";
        echo "</div>";
    } else {
        echo "<p>No records found</p>";
    }
    
    displayBackButton();
    displayFooter();
}

/**
 * Search for a specific student by ID
 */
function searchStudent($conn) {
    $search_id = $conn->real_escape_string($_GET['search_id']);
    
    $sql = "SELECT s.*, d.department_name FROM students s 
            LEFT JOIN student_department_mapping m ON s.student_id = m.student_id 
            LEFT JOIN departments d ON m.department_id = d.department_id 
            WHERE s.student_id = '$search_id'";
    
    $result = $conn->query($sql);
    
    displayHeader("Search Results");
    
    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        
        echo "<div class='student-details'>";
        echo "<h2>Student Details</h2>";
        echo "<div class='detail-row'><strong>Student ID:</strong> " . htmlspecialchars($row['student_id']) . "</div>";
        echo "<div class='detail-row'><strong>Name:</strong> " . htmlspecialchars($row['name']) . "</div>";
        echo "<div class='detail-row'><strong>Date of Birth:</strong> " . htmlspecialchars($row['dob']) . "</div>";
        echo "<div class='detail-row'><strong>Gender:</strong> " . htmlspecialchars($row['gender']) . "</div>";
        echo "<div class='detail-row'><strong>Address:</strong> " . htmlspecialchars($row['address']) . "</div>";
        echo "<div class='detail-row'><strong>Mobile:</strong> " . htmlspecialchars($row['mobile']) . "</div>";
        echo "<div class='detail-row'><strong>Email:</strong> " . htmlspecialchars($row['email']) . "</div>";
        echo "<div class='detail-row'><strong>Department:</strong> " . htmlspecialchars($row['department_name']) . "</div>";
        echo "</div>";
        
        echo "<div class='action-buttons'>";
        echo "<a href='process_form.php?action=update&id=" . $row['student_id'] . "' class='btn edit'>Edit</a>";
        echo "<a href='process_form.php?action=delete&id=" . $row['student_id'] . "' class='btn delete' onclick='return confirm(\"Are you sure you want to delete this record?\")'>Delete</a>";
        echo "</div>";
    } else {
        echo "<p>No student found with ID: " . htmlspecialchars($search_id) . "</p>";
    }
    
    displayBackButton();
    displayFooter();
}

/**
 * Fetch student data for update form
 */
function fetchStudentForUpdate($conn) {
    $student_id = $conn->real_escape_string($_GET['id']);
    
    $sql = "SELECT s.*, d.department_name FROM students s 
            LEFT JOIN student_department_mapping m ON s.student_id = m.student_id 
            LEFT JOIN departments d ON m.department_id = d.department_id 
            WHERE s.student_id = '$student_id'";
    
    $result = $conn->query($sql);
    
    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        
        displayHeader("Update Student");
        
        echo "<div class='form-container'>";
        echo "<form action='process_form.php' method='POST'>";
        echo "<input type='hidden' name='action' value='update'>";
        echo "<input type='hidden' name='original_student_id' value='" . htmlspecialchars($row['student_id']) . "'>";
        
        echo "<div class='form-group'>";
        echo "<label for='name'>Full Name:</label>";
        echo "<input type='text' id='name' name='name' value='" . htmlspecialchars($row['name']) . "' required>";
        echo "</div>";
        
        echo "<div class='form-group'>";
        echo "<label for='student_id'>Student ID:</label>";
        echo "<input type='text' id='student_id' name='student_id' value='" . htmlspecialchars($row['student_id']) . "' required>";
        echo "</div>";
        
        echo "<div class='form-group'>";
        echo "<label for='dob'>Date of Birth:</label>";
        echo "<input type='date' id='dob' name='dob' value='" . htmlspecialchars($row['dob']) . "' required>";
        echo "</div>";
        
        echo "<div class='form-group'>";
        echo "<label>Gender:</label>";
        echo "<div class='radio-group'>";
        echo "<label><input type='radio' name='gender' value='Male'" . ($row['gender'] == 'Male' ? ' checked' : '') . "> Male</label>";
        echo "<label><input type='radio' name='gender' value='Female'" . ($row['gender'] == 'Female' ? ' checked' : '') . "> Female</label>";
        echo "<label><input type='radio' name='gender' value='Other'" . ($row['gender'] == 'Other' ? ' checked' : '') . "> Other</label>";
        echo "</div>";
        echo "</div>";
        
        echo "<div class='form-group'>";
        echo "<label for='address'>Address:</label>";
        echo "<textarea id='address' name='address' rows='3' required>" . htmlspecialchars($row['address']) . "</textarea>";
        echo "</div>";
        
        echo "<div class='form-group'>";
        echo "<label for='mobile'>Mobile Number:</label>";
        echo "<input type='tel' id='mobile' name='mobile' value='" . htmlspecialchars($row['mobile']) . "' required>";
        echo "</div>";
        
        echo "<div class='form-group'>";
        echo "<label for='email'>Email ID:</label>";
        echo "<input type='email' id='email' name='email' value='" . htmlspecialchars($row['email']) . "' required>";
        echo "</div>";
        
        echo "<div class='form-group'>";
        echo "<label for='department'>Department:</label>";
        echo "<select id='department' name='department' required>";
        
        // Fetch all departments
        $dept_sql = "SELECT department_name FROM departments ORDER BY department_name";
        $dept_result = $conn->query($dept_sql);
        
        while ($dept_row = $dept_result->fetch_assoc()) {
            $selected = ($dept_row['department_name'] == $row['department_name']) ? 'selected' : '';
            echo "<option value='" . htmlspecialchars($dept_row['department_name']) . "' $selected>" . htmlspecialchars($dept_row['department_name']) . "</option>";
        }
        
        echo "</select>";
        echo "</div>";
        
        echo "<div class='btn-container'>";
        echo "<button type='submit' class='btn update'>Update Student</button>";
        echo "<a href='process_form.php?action=view' class='btn'>Cancel</a>";
        echo "</div>";
        echo "</form>";
        echo "</div>";
        
        displayFooter();
    } else {
        displayErrorMessage("Student not found!");
    }
}

/**
 * Update student information
 */
function updateStudent($conn) {
    // Sanitize and validate input data
    $original_student_id = $conn->real_escape_string($_POST['original_student_id']);
    $name = $conn->real_escape_string($_POST['name']);
    $student_id = $conn->real_escape_string($_POST['student_id']);
    $dob = $conn->real_escape_string($_POST['dob']);
    $gender = $conn->real_escape_string($_POST['gender']);
    $address = $conn->real_escape_string($_POST['address']);
    $mobile = $conn->real_escape_string($_POST['mobile']);
    $email = $conn->real_escape_string($_POST['email']);
    $department = $conn->real_escape_string($_POST['department']);
    
    // Start transaction
    $conn->begin_transaction();
    
    try {
        // Update student information
        $sql = "UPDATE students SET 
                name = ?, 
                student_id = ?, 
                dob = ?, 
                gender = ?, 
                address = ?, 
                mobile = ?, 
                email = ? 
                WHERE student_id = ?";
        
        $stmt = $conn->prepare($sql);
        $stmt->bind_param("ssssssss", $name, $student_id, $dob, $gender, $address, $mobile, $email, $original_student_id);
        
        if (!$stmt->execute()) {
            throw new Exception("Error updating student: " . $stmt->error);
        }
        
        // Check if department exists, if not, create it
        $stmt = $conn->prepare("SELECT department_id FROM departments WHERE department_name = ?");
        $stmt->bind_param("s", $department);
        $stmt->execute();
        $result = $stmt->get_result();
        
        if ($result->num_rows > 0) {
            $row = $result->fetch_assoc();
            $department_id = $row['department_id'];
        } else {
            $stmt = $conn->prepare("INSERT INTO departments (department_name) VALUES (?)");
            $stmt->bind_param("s", $department);
            $stmt->execute();
            $department_id = $conn->insert_id;
        }
        
        // Update the mapping table
        if ($original_student_id != $student_id) {
            // If student_id has changed, we need to update the mapping
            $sql = "UPDATE student_department_mapping SET student_id = ?, department_id = ? WHERE student_id = ?";
            $stmt = $conn->prepare($sql);
            $stmt->bind_param("sis", $student_id, $department_id, $original_student_id);
        } else {
            // Otherwise just update the department
            $sql = "UPDATE student_department_mapping SET department_id = ? WHERE student_id = ?";
            $stmt = $conn->prepare($sql);
            $stmt->bind_param("is", $department_id, $student_id);
        }
        
        if (!$stmt->execute()) {
            throw new Exception("Error updating mapping: " . $stmt->error);
        }
        
        // Commit transaction
        $conn->commit();
        
        displaySuccessMessage("Student updated successfully!");
        
    } catch (Exception $e) {
        // Rollback transaction on error
        $conn->rollback();
        displayErrorMessage("Error: " . $e->getMessage());
    }
}

/**
 * Delete a student record
 */
function deleteStudent($conn) {
    $student_id = $conn->real_escape_string($_GET['id']);
    
    // Start transaction
    $conn->begin_transaction();
    
    try {
        // Delete from student table (this will cascade to mapping table due to foreign key)
        $stmt = $conn->prepare("DELETE FROM students WHERE student_id = ?");
        $stmt->bind_param("s", $student_id);
        
        if (!$stmt->execute()) {
            throw new Exception("Error deleting student: " . $stmt->error);
        }
        
        // Commit transaction
        $conn->commit();
        
        displaySuccessMessage("Student deleted successfully!");
        
    } catch (Exception $e) {
        // Rollback transaction on error
        $conn->rollback();
        displayErrorMessage("Error: " . $e->getMessage());
    }
}

/**
 * Display HTML header
 */
function displayHeader($title) {
    echo "<!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>$title</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #2c3e50;
            }
            .message {
                padding: 10px;
                margin-bottom: 20px;
                border-radius: 4px;
                text-align: center;
            }
            .success {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .table-container {
                overflow-x: auto;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #3498db;
                color: white;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .btn {
                display: inline-block;
                padding: 8px 12px;
                margin: 5px;
                text-decoration: none;
                color: white;
                border-radius: 4px;
                text-align: center;
                cursor: pointer;
            }
            .back {
                background-color: #3498db;
            }
            .edit {
                background-color: #f39c12;
            }
            .delete {
                background-color: #e74c3c;
            }
            .update {
                background-color: #27ae60;
            }
            .btn-container {
                text-align: center;
                margin-top: 20px;
            }
            .form-container {
                max-width: 700px;
                margin: 0 auto;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type='text'],
            input[type='email'],
            input[type='date'],
            input[type='tel'],
            select,
            textarea {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            .radio-group {
                display: flex;
                gap: 20px;
            }
            .student-details {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .detail-row {
                margin-bottom: 10px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .action-buttons {
                text-align: center;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class='container'>
            <h1>$title</h1>";
}

/**
 * Display footer
 */
function displayFooter() {
    echo "</div>
    </body>
    </html>";
}

/**
 * Display success message
 */
function displaySuccessMessage($message) {
    displayHeader("Operation Successful");
    echo "<div class='message success'>$message</div>";
    displayBackButton();
    displayFooter();
}

/**
 * Display error message
 */
function displayErrorMessage($message) {
    displayHeader("Error");
    echo "<div class='message error'>$message</div>";
    displayBackButton();
    displayFooter();
}

/**
 * Display back button
 */
function displayBackButton() {
    echo "<div class='btn-container'>";
    echo "<a href='student_form.html' class='btn back'>Back to Form</a>";
    echo "<a href='process_form.php?action=view' class='btn'>View All Students</a>";
    echo "</div>";
}
?>
// Tab navigation
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show the selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked button
    document.querySelector(`.tab-btn[onclick="showTab('${tabName}')"]`).classList.add('active');
}

// Validation functions
function validateName() {
    const nameInput = document.getElementById('name');
    const nameError = document.getElementById('nameError');
    const nameRegex = /^[a-zA-Z\s]+$/;
    
    if (nameInput.value.trim() === "") {
        nameError.textContent = "Name cannot be blank";
        nameError.style.display = "block";
        return false;
    } else if (!nameRegex.test(nameInput.value)) {
        nameError.textContent = "Name should contain only alphabets and spaces";
        nameError.style.display = "block";
        return false;
    } else {
        nameError.style.display = "none";
        return true;
    }
}

function validateDOB() {
    const dobInput = document.getElementById('dob');
    const dobError = document.getElementById('dobError');
    const dobRegex = /^(0[1-9]|[12][0-9]|3[01])\/(0[1-9]|1[0-2])\/\d{4}$/;
    
    if (dobInput.value.trim() === "") {
        dobError.textContent = "Date of Birth cannot be blank";
        dobError.style.display = "block";
        return false;
    } else if (!dobRegex.test(dobInput.value)) {
        dobError.textContent = "Date must be in dd/mm/yyyy format";
        dobError.style.display = "block";
        return false;
    } else {
        dobError.style.display = "none";
        calculateAge();
        return true;
    }
}

function calculateAge() {
    const dobInput = document.getElementById('dob');
    const ageInput = document.getElementById('age');
    
    if (dobInput.value) {
        const parts = dobInput.value.split('/');
        if (parts.length === 3) {
            const birthDate = new Date(parts[2], parts[1] - 1, parts[0]);
            const today = new Date();
            let age = today.getFullYear() - birthDate.getFullYear();
            const monthDiff = today.getMonth() - birthDate.getMonth();
            
            if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
                age--;
            }
            
            ageInput.value = age;
        }
    } else {
        ageInput.value = '';
    }
}

function validateMobile() {
    const mobileInput = document.getElementById('mobile');
    const mobileError = document.getElementById('mobileError');
    const mobileRegex = /^[1-9][0-9]*$/;
    
    if (mobileInput.value.trim() === "") {
        mobileError.textContent = "Mobile number cannot be blank";
        mobileError.style.display = "block";
        return false;
    } else if (!mobileRegex.test(mobileInput.value)) {
        mobileError.textContent = "Mobile number must contain only digits and cannot start with 0";
        mobileError.style.display = "block";
        mobileInput.value = "";
        return false;
    } else {
        mobileError.style.display = "none";
        return true;
    }
}

function validateEmail() {
    const emailInput = document.getElementById('email');
    const emailError = document.getElementById('emailError');
    
    if (emailInput.value.trim() === "") {
        emailError.textContent = "Email cannot be blank";
        emailError.style.display = "block";
        return false;
    }
    
    const email = emailInput.value;
    
    if (!email.includes('@') || email.startsWith('@') || 
        !(email.endsWith('.com') || email.endsWith('.in'))) {
        emailError.textContent = "Email must contain '@' and end with '.com' or '.in'";
        emailError.style.display = "block";
        emailInput.value = "";
        return false;
    } else {
        emailError.style.display = "none";
        return true;
    }
}

// Student Class
class Student {
    constructor(name, dob, age, mobile, email) {
        this.name = name;
        this.dob = dob;
        this.age = age;
        this.mobile = mobile;
        this.email = email;
    }
}

// Cookie functions
function setCookie(name, value, days) {
    const expires = new Date();
    expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
    document.cookie = name + '=' + JSON.stringify(value) + ';expires=' + expires.toUTCString() + ';path=/';
}

function getCookie(name) {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === ' ') c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) === 0) return JSON.parse(c.substring(nameEQ.length, c.length));
    }
    return null;
}

// Form submission
function submitForm() {
    const isNameValid = validateName();
    const isDOBValid = validateDOB();
    const isMobileValid = validateMobile();
    const isEmailValid = validateEmail();
    
    if (isNameValid && isDOBValid && isMobileValid && isEmailValid) {
        const name = document.getElementById('name').value;
        const dob = document.getElementById('dob').value;
        const age = document.getElementById('age').value;
        const mobile = document.getElementById('mobile').value;
        const email = document.getElementById('email').value;
        
        // Create Student object
        const student = new Student(name, dob, age, mobile, email);
        
        // Store in cookie
        setCookie('studentData_' + mobile, student, 30);
        
        // Clear form and show success message
        document.getElementById('studentForm').reset();
        alert("Application submitted successfully! Your data has been saved.");
        
        // Switch to the retrieve tab
        showTab('retrieve');
        document.getElementById('searchMobile').value = mobile;
    }
}

// Retrieve cookie data
function retrieveCookie() {
    const mobileNumber = document.getElementById('searchMobile').value;
    
    if (!mobileNumber) {
        alert('Please enter a mobile number');
        return;
    }
    
    const studentData = getCookie('studentData_' + mobileNumber);
    
    if (studentData) {
        let message = "Student Details:\n\n";
        message += "Name: " + studentData.name + "\n";
        message += "Date of Birth: " + studentData.dob + "\n";
        message += "Age: " + studentData.age + "\n";
        message += "Mobile: " + studentData.mobile + "\n";
        message += "Email: " + studentData.email;
        
        alert(message);
    } else {
        alert('No data found for this mobile number');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('dob').addEventListener('input', function() {
        // Format the input as the user types
        let value = this.value;
        value = value.replace(/\D/g, '');
    
        if (value.length > 8) {
            value = value.substring(0, 8);
        }
    
        if (value.length > 4) {
            value = value.substring(0, 4) + '/' + value.substring(4);
        }
        
        if (value.length > 2) {
            value = value.substring(0, 2) + '/' + value.substring(2);
        }
        
        this.value = value;
    });
});
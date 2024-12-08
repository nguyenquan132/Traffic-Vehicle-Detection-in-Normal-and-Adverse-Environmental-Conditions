
let sign_in = document.querySelector('.sign');

function log_in(){
    // Ngăn chặn hành vi mặc định của form
    document.getElementById('loginForm').addEventListener('submit', function(event) { 
        event.preventDefault(); });
    window.location.href = 'traffic_vehicle.html';
}

sign_in.addEventListener('click', log_in)
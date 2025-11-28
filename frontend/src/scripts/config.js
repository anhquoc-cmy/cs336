window.BACKEND_BASE = 'http://localhost:8000';   // REST API base
window.WS_URL = (window.BACKEND_BASE).replace(/^http/, 'ws') + '/ws';
window.USE_WS = false; // Disable legacy sockets by default
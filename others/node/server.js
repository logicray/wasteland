var http = require('http')

http.createServer(function(request, response) {
    //send http header
	//http status code
	//
	response.writeHead(200, {'Content-Type':'text/plain'});

	response.end('Hello, world\n');
}).listen(8989);

console.log('Server running at http://127.0.0.1:8989/')

 require(['base/js/namespace', 'base/js/events'], function(IPython, events) {
     events.on('notebook_loaded.Notebook', function() {
        console.log('require & notebook_loaded.Notebook');
        init();
    });
     events.on('app_initialized.NotebookApp', function() {
        console.log('require & app_initialized.NotebookApp');
        init();
     });
 });

 define(['base/js/namespace', 'base/js/events'], function(IPython, events) {
     events.on('notebook_loaded.Notebook', function() {
        console.log('define & notebook_loaded.Notebook');
        init();
    });
     events.on('app_initialized.NotebookApp', function() {
         console.log('define & app_initialized.NotebookApp');
         init();
         
     });
 });
 
 
 require(['base/js/namespace', 'base/js/events'], function(IPython, events) {
	    console.log("A");
	    events.on('app_initialized.NotebookApp', function() {
	        console.log("B");
	    });
	    console.log("C");
	});
 
 test = "test";
 
 
 function init() {
	 console.log("here3");
 }
 
 
 console.log("C");
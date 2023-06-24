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

 
 function init() {
	 $(".dynamic-instructions:contains('Select items to perform actions on them.')").text("");
	 $("a:contains('Control Panel')").html("<i class='fa fa-tasks'></i>");
	 $("#logout").html("<i class='fa fa-times'></i>")
	 // Swap upload text with icon
	 $(".btn-upload").html("<input title='' type='file' name='datafile' class='fileinput' multiple='multiple'><i class='fa fa-cloud-upload'></i>")
	 // Swap new text with icon
	 $("span:contains('New')").html("<i class='fa fa-plus-square-o'></i>") 
	 if (!document.getElementById) {
		 document.write('<link rel="stylesheet" type="text/css" href="/home/alec/.jupyter/custom/css/font-awesome.min.css">');
	 }

	 
 }
 
 
 console.log("C");
 init();
function fileSelect(obj) {
    var img = document.getElementById("before");
    let file = obj.files[0];
    let reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function(){
        img.setAttribute("src", this.result);
    }
    changeStatus("选择: " + file.name);
    $('#after').attr("src", "");
    $('#result tbody').html("");
};

function showTable(data) {
    var txt = "";

    for(var i in data) {
        txt += "<tr>"
        var item = data[i];
        txt += "<td>" + item.label + "</td>"
             + "<td>" + item.xmin + "</td>"
             + "<td>" + item.ymin + "</td>"
             + "<td>" + item.xmax + "</td>"
             + "<td>" + item.ymax + "</td>"
             + "<td>" + item.confidence + "</td>";
        txt += "</tr>";
    }
    $('#result tbody').html("");
    if(txt != "") {
        $("#result").append(txt).removeClass("hidden");
    }
}

function changeStatus(str) {
    $('#status').attr("value", str);
}

$('#btnFileUpload').click(function () {
    var formFile = new FormData($('#formUpload')[0])
    changeStatus("图片上传中...");
    // console.log("Get file success");
    $.ajax({
        url: "/detect",
        type: "POST",
        data: formFile,
        processData: false,
        contentType: false,
        success: function (response) {
            if(response.status == 1) {
                // console.log("success");
                changeStatus("检测成功...推理时间: " + response.time);
                $('#after').attr("src", response.result_url);
                var data = response.img_info;
                showTable(data);
            }
            if(response.status == 2) {
                // console.log("detect nothing");
                changeStatus("未检测到目标...");
                $('#result tbody').html("");
            }
        },
        error: function(response) {
            alert("上传失败")
        }
    });
});

function ajaxCall(paramData, callBack){
    let method = paramData.method
    let url = paramData.url
    let data = paramData.data
    $.ajax({
        type: method,
        url: url,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
        data: data,
        success: function( data ) {
            callBack(data)
        },
        error: function (response) {
            alert(response)
            console.log(response)
        }
    });
}
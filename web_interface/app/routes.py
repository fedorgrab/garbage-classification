from app import app
from flask import render_template, request, send_from_directory, Response
from app.classifier import GarbageClassifier
from config import ClassifierConfig

garbage_classifier = GarbageClassifier()


@app.route("/")
def upload():
    return render_template("file_upload_form.html")


@app.route("/success", methods=["POST"])
def success():
    image_file = request.files["file"]

    if not image_file.filename.endswith(".jpg"):
        return Response(
            "Incorrect Image format. It should be in jpg "
            "and the image file name should ends with `.jpg`"
        )

    predicted_class, image_file_path = garbage_classifier.classify_image(image_file=image_file)
    same_sample_names = garbage_classifier.get_same_sample_names(predicted_class)

    return render_template(
        template_name_or_list="success.html",
        class_name=predicted_class,
        classified_image=image_file_path,
        same_sample_names=same_sample_names,
    )


@app.route("/tested-garbage/<path:filename>", methods=["GET"])
def get_data_file(filename):
    return send_from_directory(
        directory=ClassifierConfig.UPLOADED_IMAGES_FILE_PATH, filename=filename,
    )

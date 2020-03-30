from flask import Response, render_template, request, send_from_directory

from web_interface.app import app
from web_interface.app.classifier import GarbageClassifier
from web_interface.config import ClassifierConfig

ALLOWED_FILE_EXTENSIONS = ("jpg", "jpeg", "HEIC")
garbage_classifier = GarbageClassifier()


@app.route("/")
def upload():
    return render_template("file_upload_form.html")


@app.route("/success", methods=["POST"])
def success():
    image_file = request.files["file"]
    file_extension = image_file.filename.split(".")[-1]

    if file_extension not in ALLOWED_FILE_EXTENSIONS:
        return Response(
            f"Incorrect Image format. File extension should be one of the provided:"
            f" {', '.join(ALLOWED_FILE_EXTENSIONS)} "
        )

    predicted_class, image_file_path = garbage_classifier.classify_image(
        image_file=image_file
    )
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
        directory=ClassifierConfig.UPLOADED_IMAGES_FILE_PATH, filename=filename
    )

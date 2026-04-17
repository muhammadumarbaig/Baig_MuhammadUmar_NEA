from flask import Flask, render_template, request, redirect, jsonify, url_for, session, flash, send_file
from os import path
import shutil
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import base64
from werkzeug.utils import secure_filename
from datetime import datetime
import re, joblib
import pandas as pd
import numpy as np
from hvac_model_training import (
    DataSet, 
    SensorBiasClassifier, 
    add_temperature_deltas, 
    create_stability_features
)
app = Flask(__name__)

db_locale = "DATABASE1.db"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
#app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'RMS>=AM>=GM>=HM'
if not path.exists(db_locale):
    db_locale = db_locale
    con = sqlite3.connect(db_locale)
    con.execute('PRAGMA foreign_keys = ON;')
    command = con.cursor()

    # creates org table
    command.execute("""CREATE TABLE IF NOT EXISTS Org(
        "OrgID" INTEGER,
        "OrgName" TEXT,
        "OrgImage" BLOB,
        PRIMARY KEY("OrgID")
    );""")
    con.commit()

    # creates users table
    command.execute("""CREATE TABLE IF NOT EXISTS Users(
        "UserID" INTEGER,
        "OrgID" INTEGER,
        "Username" TEXT UNIQUE,
        "PasswordHash" TEXT,
        PRIMARY KEY("UserID"),
        FOREIGN KEY("OrgID") REFERENCES "Org"("OrgID")
    );""")
    con.commit()

    # creates site table
    command.execute("""CREATE TABLE IF NOT EXISTS Site(
        "SiteID" INTEGER,
        "OrgID" INTEGER,
        "SiteName" TEXT NOT NULL,
        "SiteImage" BLOB,
        PRIMARY KEY("SiteID"),
        FOREIGN KEY("OrgID") REFERENCES "Org"("OrgID")
    );""")
    con.commit()

    # creates site users table
    command.execute("""CREATE TABLE IF NOT EXISTS SiteUsers(
        "UserID" INTEGER,
        "SiteID" INTEGER,
        "AdminStatus" INTEGER,
        PRIMARY KEY("UserID","SiteID"),
        FOREIGN KEY("SiteID") REFERENCES "Org"("OrgID"),
        FOREIGN KEY("UserID") REFERENCES "Users"("UserID")
    );""")
    con.commit()

    # creates files table
    command.execute("""CREATE TABLE IF NOT EXISTS Files(
        "FileID" INTEGER,
        "FileExtension" TEXT,
        "SiteID" INTEGER,
        "DateTime" TEXT,
        "FileName" TEXT,
        PRIMARY KEY("FileID"),
        FOREIGN KEY("SiteID") REFERENCES "Site"("SiteID")
    );""")
    con.commit()

    # creates processed files table
    command.execute("""CREATE TABLE IF NOT EXISTS ProcessedFiles(
        "FileID" INTEGER,
        "ModelUsed" TEXT,
        "Subject" TEXT,
        FOREIGN KEY("FileID") REFERENCES "Files"("FileID")
    );""")
    con.commit()

    con.close()
def insertorg(name , file_path):
    with open(file_path, "rb") as file:
        filedata = file.read()
    con = sqlite3.connect(db_locale)
    command = con.cursor()
    sqlstring = """INSERT INTO Org (OrgName, OrgImage) VALUES (?, ?)"""
        
    command.execute(sqlstring, (name, filedata))
    
    con.commit()
    con.close()
def get_user_id(username):
    con = sqlite3.connect(db_locale)
    command = con.cursor()
    command.execute("""SELECT UserID FROM Users WHERE Username = ?""", (username,))
    return command.fetchone()[0]
def get_org_id(name):

    con = sqlite3.connect(db_locale)
    command = con.cursor()
    command.execute("""SELECT OrgID FROM Org WHERE OrgName = ?""", (name,))
    return command.fetchone()[0]
def get_site_id(sitename):
    con = sqlite3.connect(db_locale)
    command = con.cursor()
    command.execute("""SELECT SiteID FROM Site WHERE SiteName = ?""", (sitename,))
    return command.fetchone()[0]

def insertsite(orgname, sitename, file_path):
    orgid = get_org_id(orgname)
    with open(file_path, "rb") as file:
        filedata = file.read()

    con = sqlite3.connect(db_locale)
    command = con.cursor()
    sqlstring = """INSERT INTO Site (OrgID, SiteName, SiteImage) VALUES (?, ?, ?)"""
    command.execute(sqlstring, (orgid, sitename, filedata))
    con.commit()
    con.close()

#__________________ADDING ORGS function______________________#
def addorgs():
    #imgdir = r"C:\Users\ADMIN\OneDrive - Riverside College Halton\organdsiteimages"
    base_path = path.dirname(__file__)
    imgdir = path.join(base_path, "organdsiteimages")
    insertorg("CERN", rf"{imgdir}\CERNLOGO1.png")
    insertorg("STFC", rf"{imgdir}\STFCLOGOIMAGE.jfif")
    insertorg("Lawrence Berkeley", rf"{imgdir}\BerkeleyLabLogo.png")
#__________________ADDING SITES function_____________________#
def addsites():
    #imgdir = r"C:\Users\ADMIN\OneDrive - Riverside College Halton\organdsiteimages"
    base_path = path.dirname(__file__)
    imgdir = path.join(base_path, "organdsiteimages")
    insertsite("CERN", "Prevessin-Moens", rf"{imgdir}\Prevessinimage.png")
    insertsite("CERN", "Meyrin", rf"{imgdir}\MeyrinSiteImage.jfif")
    insertsite("STFC", "Daresbury", rf"{imgdir}\ukri-stfc-logo.png")
    insertsite("STFC", "Rutherford Appleton Lab", rf"{imgdir}\Rutherfordappletonimage.jpg")
    insertsite("Lawrence Berkeley", "Advanced Light Source", rf"{imgdir}\AdvancedLightSource.png")
    insertsite("Lawrence Berkeley", "Molecular Foundry", rf"{imgdir}\MolecularFoundry.jpg")
    insertsite("Lawrence Berkeley", "Joint Genome Institute", rf"{imgdir}\JointGenomeInstitute.png")
#___________________BOOTSTRAP DATA____________________________
def bootstrap_data():
    con = sqlite3.connect(db_locale)
    command = con.cursor()
    command.execute("""SELECT COUNT(*) FROM Org""")
    if command.fetchone()[0] == 0:
        addorgs()
        addsites()
    con.close()

bootstrap_data()


# HVAC FILE PROCESSING


class HVACFileProcessor:
    
    def __init__(self, classifier=None):
        if classifier is None:
            self.classifier = self._load_pretrained_classifier()
        else:
            self.classifier = classifier
        
    @staticmethod
    def get_user_submitted_file(file_id):
        """Load a user-submitted CSV file from the FILES folder."""
        try:
            con = sqlite3.connect(db_locale)
            command = con.cursor()
            command.execute("""SELECT FileExtension FROM Files WHERE FileID = ?""", (file_id,))
            result = command.fetchone()
            con.close()
            
            if result is None:
                raise FileNotFoundError("File not found in database")
            
            file_extension = result[0]
            file_path = path.join(path.dirname(__file__), "FILES")
            file_path = path.join(file_path, f"{file_id}{file_extension}")
            
            if not path.exists(file_path):
                raise FileNotFoundError("File not found in computer")
            
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load file: {e}")

    @staticmethod
    def validate_user_file(df):

        required_columns = [
            "Datetime", "RMCLGSPT", "RMHTGSPT", "FCU_CVLV_DM", "FCU_HVLV_DM",
            "FCU_DA_CFM", "FCU_SPD", "FCU_MAT", "FCU_DAT", "RM_TEMP",
            "FCU_CVLV", "FCU_HVLV", "FAN_CTRL"
        ]
        #
        missing_columns = []
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
            else:
                pass

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
        
#missing_columns = [col for col in required_columns if col not in df.columns]
    @staticmethod
    def process_user_file(df):
        """Filter by operating hours and reduce to important parameters."""
        try:
            # Filter to operating hours only (weekdays 1-5, hours 6-17)
            processed_df = DataSet.conv_op_hrs_only(df)
            
            # Reduce to important parameters
            processed_df = DataSet.ReduceToImportantParamsOnly(processed_df)
            
            return processed_df
        except Exception as e:
            raise RuntimeError(f"Failed to process file: {e}")

    @staticmethod
    def add_engineered_features(df):
        try:
            df = df.copy()
            
            # Add temperature delta features
            add_temperature_deltas(df)
            
            # Add stability features
            create_stability_features([df])
            
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to add engineered features: {e}")

    def predict_bias_labels(self, df):
        """Predict sensor bias labels for each row using trained model."""
        try:
            if self.classifier is None or self.classifier.model is None:
                raise RuntimeError("Classifier not loaded")
            df = df.copy()
            
            # Get feature columns the model expects (from saved classifier if available)
            if hasattr(self.classifier, "feature_columns") and self.classifier.feature_columns:
                feature_columns = self.classifier.feature_columns
            else:
                feature_columns = [
                    "DELTA_COOLINGSETPT",
                    "FCU_CVLV",
                    "FCU_CVLV_DM",
                    "FCU_DAT",
                    "FCU_MAT",
                    "RM_TEMP",
                    "FCU_DA_CFM",
                    "IsInStabilityZone"
                ]
            
            # Verify all features are present
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Get feature data
            X = df[feature_columns].values
            
            # Get predictions
            predictions = self.classifier.model.predict(X)
            probabilities = self.classifier.model.predict_proba(X)
            
            # Add predictions to dataframe
            df['IsBiased'] = predictions
            df['PredictionConfidence'] = probabilities.max(axis=1)
        
            return df
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def process_and_label_file(self, file_id):
        # Step 1: Load file
        df = self.get_user_submitted_file(file_id)
        
        # Step 2: Validate file
        self.validate_user_file(df)
        
        # Step 3: Process file (filter operating hours, reduce params)
        df = self.process_user_file(df)
        if df is None or len(df) == 0:
            raise ValueError("No data after filtering to operating hours (weekdays 6-17)")
        
        # Step 4: Add engineered features
        df = self.add_engineered_features(df)
        
        # Step 5: Make predictions
        df = self.predict_bias_labels(df)
        
        return df

    @staticmethod
    def save_processed_file(df, file_id):
        """Save processed and labeled file back to FILES folder."""
        try:
            base_path = path.join(path.dirname(__file__), "FILES")
            output_path = path.join(base_path, f"{file_id}_processed.csv")
            df.to_csv(output_path, index=False)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to save processed file: {e}")

    @staticmethod
    def _load_pretrained_classifier(pkldirname = "sensor_bias_classifier.pkl"):
        """Load pre-trained classifier from pkl."""
        classifier_path = path.join(path.dirname(__file__), pkldirname)
        
        # Try to load existing model
        if path.exists(classifier_path):
            try:
                classifier = joblib.load(classifier_path)
                return classifier
            except Exception:
                pass

        return None


# Initialize processor on app startup
HVAC_PROCESSOR = HVACFileProcessor()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        userorg = request.form["userorg"]
        return redirect(url_for("orgpage", userorg=userorg))

    else:    
        con = sqlite3.connect(db_locale)
        command = con.cursor()
        command.execute("""SELECT OrgName FROM Org""")
        allorgs_tuple = command.fetchall()
        con.close()
        allorgs= []
        for org_tuple in allorgs_tuple:
            allorgs.append(org_tuple[0])
        return render_template("index.html", allorgs = allorgs)

def get_org_image_blob(orgname):
    con = sqlite3.connect(db_locale)
    command = con.cursor()
    command.execute("""SELECT OrgImage FROM Org WHERE OrgName = ?""", (orgname,))
    blob = command.fetchone()[0]
    con.close()
    return blob

def get_site_image_blob(sitename):
    con = sqlite3.connect(db_locale)
    command = con.cursor()
    command.execute("""SELECT SiteImage FROM Site WHERE SiteName = ?""", (sitename,))
    blob = command.fetchone()[0]
    con.close()
    return blob

def decode_blob(blob):
    encoded_blob = base64.b64encode(blob).decode("utf-8")
    mime_type="image/png"
    imgdata = f"data:{mime_type};base64,{encoded_blob}"
    return imgdata

@app.route("/login/<userorg>", methods=["GET", "POST"])
def orgpage(userorg):
    if request.method == "POST":
        username = request.form["username"]
        entered_password = request.form["password"]
        

        con =sqlite3.connect(db_locale)
        command = con.cursor()
        command.execute("""SELECT Username FROM Users WHERE Username = ?""", (username,))
        if command.fetchone() is None:
            verified = False
        else:
            command.execute("""SELECT PasswordHash FROM Users WHERE Username = ?""", (username,))
            password_hash = command.fetchone()[0]
            verified = check_password_hash(password_hash, entered_password)
        if verified:
            userid = get_user_id(username)
            con = sqlite3.connect(db_locale)
            command = con.cursor()
            command.execute("""SELECT Site.SiteName FROM Site INNER JOIN SiteUsers
                           ON Site.SiteID = SiteUsers.SiteID INNER JOIN Users
                           ON SiteUsers.UserID = Users.UserID
                           WHERE Users.UserID = ?""", (userid,))
            sitename = command.fetchone()[0]
            return redirect(url_for("sitepage", username=username, sitename=sitename))
        else:
            flash("Login Failed! Please check your username and password.")
        return redirect(url_for("orgpage", userorg=userorg))
    else:
        blob = get_org_image_blob(userorg)
        imgdata = decode_blob(blob)
        session["userorg"] = userorg
        return render_template("loginorg.html", userorg=userorg, img = imgdata)
class FileHandler():
    def __init__(self, rawfilename = None, foldername = None):
        if not rawfilename is None or foldername is None:
            self.rawfilename = rawfilename
            self.foldername = foldername
            self.securefilename = secure_filename(self.rawfilename)
            strip = self.securefilename.split(".")
            self.strpfilename = strip[0]
            self.extension = "." + strip[1]
    def VerifyExtension(self):
        allowed_extensions = [".xml", ".xlsx", ".csv"]
        if self.extension in allowed_extensions:
            return True
        else:
            return False
    def InsertFileDataToDatabase(self, siteid, date_time):
        con = sqlite3.connect(db_locale)
        command = con.cursor()
        sqlstring = ("""INSERT INTO Files
                        ("FileExtension", "FileName", "SiteID", "DateTime")
                        VALUES (?, ?, ?, ?)""")
        command.execute(sqlstring, (self.extension, self.strpfilename, siteid, date_time))
        con.commit()
        sqlstring = ("""SELECT FileID FROM Files WHERE FileExtension = ? AND FileName = ?
                     AND SiteID = ? AND DateTime = ?""")
        command.execute(sqlstring, (self.extension, self.strpfilename, siteid, date_time))
        self.fileid = command.fetchone()[0]

    def FileSavingPath(self):
        base_path = path.join(path.dirname(__file__), self.foldername)
        self.filesavingpath = path.join(base_path, str(self.fileid)+self.extension)
    @staticmethod
    def DoesFileExistInSystemForSite(siteid):
        con = sqlite3.connect(db_locale)
        command = con.cursor()
        command.execute("""SELECT COUNT(*) FROM Files WHERE SiteID = ?""", (siteid,))
        if command.fetchone()[0] == 0:
            return False
        else:
            return True
    @staticmethod
    def ValidatePassword(pwd):
        """Minimum 8 Characters Necessary.
        At least one capital.
        At least one lowercase.
        At least one digit.
        At least one special character."""
        return re.search(r"(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&.])(^[A-Za-z0-9@$!%*?&.]{8,30}$)", pwd)
    @staticmethod
    def ValidateUsername(usr):
        #must be total 2 to 30 characters
        #must start with word charac but no underscre
        #no consecutive special characters
        #must not end with special character
        return re.search(r"^(?!.*[._-]{2})(?=^.{2,30}$)[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$", usr)

@app.route("/register/<userorg>", methods=["GET", "POST"])
def register(userorg):
    blob = get_org_image_blob(userorg)
    imgdata = decode_blob(blob)
    orgid = get_org_id(userorg)
    con = sqlite3.connect(db_locale)
    command = con.cursor()
    command.execute("""SELECT SiteName FROM Site WHERE OrgID=?""", (orgid,))
    allsites = [sitetuple[0] for sitetuple in command.fetchall()]
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        usersite = request.form["usersite"]
        siteid = get_site_id(usersite)
        confirm_password = request.form["confirm_password"]
        user_password_hash = generate_password_hash(password)
        if password == confirm_password:

            con = sqlite3.connect(db_locale)
            command = con.cursor()
            #check if username already exists in db
            command.execute("""SELECT Username FROM Users WHERE Username = ?""", (username,))
            if command.fetchone() is not None:
                flash("This Username already exists!")
                return redirect(url_for("register", userorg=userorg))
                
            input_is_clean = True

            if not FileHandler.ValidateUsername(username):
                flash("Username Min 2 Characters.\nStart with word character\nAt least one lowercase.\nNo consec. special charac.s\nDoes not end with special character.")
                input_is_clean = False
            if not FileHandler.ValidatePassword(password):
                flash("Password must be Min 8 Characters.\nAt least capital\nAt least one lowercase\nAt least one digit\nAt least one special character")
                input_is_clean = False

            if not input_is_clean:
                return redirect(url_for("register", userorg=userorg))
        
            else:
                con = sqlite3.connect(db_locale)
                command = con.cursor()

                sqlstring = """INSERT INTO Users(OrgID, Username, PasswordHash) VALUES(?, ?, ?)"""
                command.execute(sqlstring, (orgid, username, user_password_hash))
                con.commit()

                userid = get_user_id(username)
                sqlstring = """INSERT INTO SiteUsers(UserID, SiteID, AdminStatus) VALUES(?, ?, ?)"""
                command.execute(sqlstring, (userid, siteid, 0))
                con.commit()
                return redirect(url_for("orgpage", userorg = userorg))
        else:
            flash("Password entered does not match!")
            return redirect(url_for("register", userorg=userorg))
    else:
        return render_template("register.html", userorg=userorg, img = imgdata, allsites=allsites)


@app.route("/download/<file_id>")
def download_file(file_id):
    processed_file_path = path.join("FILES", f"{file_id}_processed.csv")
    if path.exists(processed_file_path):
        return send_file(processed_file_path, as_attachment=True, download_name=f"hvac_processed_{file_id}.csv")
    else:
        flash("Processed file not found.")
        return redirect(url_for("index"))
        

@app.route("/<username>/<sitename>", methods=["GET", "POST"])
def sitepage(username, sitename):
    blob = get_site_image_blob(sitename)
    imgdata = decode_blob(blob)
    siteid = get_site_id(sitename)
    #filesubmitted = FileHandler.DoesFileExistInSystemForSite(siteid)
    if request.method == "POST":
        uploaded_file = request.files.get("newfile")
        if uploaded_file is not None and uploaded_file.filename is not None:
            # UPLOAD NEW FILE
            handledfile = FileHandler(uploaded_file.filename, "FILES")
            if not handledfile.VerifyExtension():
                flash("Only .xml, .xlsx, or .csv files are allowed.")
                return redirect(url_for("sitepage", username=username, sitename=sitename))
            date_time = datetime.now()
            handledfile.InsertFileDataToDatabase(siteid, date_time)
            handledfile.FileSavingPath()
            uploaded_file.save(handledfile.filesavingpath)
            
            # Process the uploaded file
            try:
                processed_df = HVAC_PROCESSOR.process_and_label_file(handledfile.fileid)
                output_path = HVAC_PROCESSOR.save_processed_file(processed_df, handledfile.fileid)
                
                # Record in ProcessedFiles table
                con = sqlite3.connect(db_locale)
                command = con.cursor()
                command.execute(
                    """INSERT INTO ProcessedFiles (FileID, ModelUsed, Subject) VALUES (?, ?, ?)""",
                    (handledfile.fileid, "LogisticRegression", "Sensor Bias Detection")
                )
                con.commit()
                con.close()
                
                flash(f"File uploaded and processed successfully! File ID: {handledfile.fileid}")
            except Exception as e:
                flash(f"File uploaded but processing failed: {str(e)}")
            
            return redirect(url_for("sitepage", username=username, sitename=sitename, last_processed=handledfile.fileid))
        else:
            # SELECT OLD FILE FROM DROPDOWN
            selected_file = request.form.get("oldfile")
            if not selected_file:#if is None (blank submitted)
                flash("Please choose a file to upload or select from the dropdown.")
                return redirect(url_for("sitepage", username=username, sitename=sitename))
            con = sqlite3.connect(db_locale)
            command = con.cursor()
            command.execute(
                """SELECT FileID, FileExtension FROM Files
                   WHERE FileName = ?
                   ORDER BY DateTime DESC""",
                (selected_file,),
            )
            row = command.fetchone()
            con.close()
            if row is None:
                flash("Selected file not found.")
                return redirect(url_for("sitepage", username=username, sitename=sitename))

            source_fileid, extension = row
            source_path = path.join("FILES", f"{source_fileid}{extension}")
            if not path.exists(source_path):
                flash("Selected file not found on disk.")
                return redirect(url_for("sitepage", username=username, sitename=sitename))

            if not FileHandler(f"{selected_file}{extension}", "FILES").VerifyExtension():
                flash("Only .xml, .xlsx, or .csv files are allowed.")
                return redirect(url_for("sitepage", username=username, sitename=sitename))

            date_time = datetime.now()
            con = sqlite3.connect(db_locale)
            command = con.cursor()
            command.execute(
                """UPDATE Files
                   SET DateTime = ?
                   WHERE FileID = ?""",
                (date_time, source_fileid),
            )
            con.commit()
            con.close()
            
            flash("File selected successfully!")
            return redirect(url_for("sitepage", username=username, sitename=sitename, last_processed=source_fileid))
    else:
        filesubmitted = FileHandler.DoesFileExistInSystemForSite(siteid)
        last_processed = request.args.get("last_processed")
        
        if filesubmitted:
            flash(f"Would you like to submit HVAC data for {sitename} again?")
            con = sqlite3.connect(db_locale)
            command = con.cursor()
            command.execute("""SELECT FileName FROM Files ORDER BY DateTime DESC""")
            all_file_names_tuple = command.fetchall()
            all_file_names = []
            for file_name_tuple in all_file_names_tuple:
                all_file_names.append(file_name_tuple[0])
            con.close()

            return render_template("site.html", username=username, sitename=sitename, img = imgdata, submitted=True, files = all_file_names, last_processed=last_processed)
        else:
            return render_template("site.html", username=username, sitename=sitename, img = imgdata, submitted=False)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

#use this option if doubling errors: use_reloader=False








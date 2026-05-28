plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.9.1"));
    testImplementation("org.junit.jupiter:junit-jupiter");

    implementation("org.apache.commons:commons-lang3:3.14.0")
    implementation("org.apache.pdfbox:pdfbox:3.0.2");
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<Jar> {
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    manifest {
        attributes["Main-Class"] =  "org.suas.aidetect.Main"
    }
    configurations["compileClasspath"].forEach { file: File ->
        from(zipTree(file.absoluteFile))
    }
}
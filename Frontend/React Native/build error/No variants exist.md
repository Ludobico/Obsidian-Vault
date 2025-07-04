- [[#How to solve|How to solve]]


이 에러는 [[Build react native expo]] 에서 <font color="#ffff00">eas build</font> 를 실행하는 과정에서 프로젝트의 빌드가 실패했을때 발생하며, 전체 에러 로그는 다음과 같습니다.

```
FAILURE: Build failed with an exception.
* What went wrong:
Could not determine the dependencies of task ':app:lintVitalReportRelease'.
> Could not resolve all dependencies for configuration ':app:releaseCompileClasspath'.
   > Could not resolve project :react-native-async-storage_async-storage.
     Required by:
         project :app
      > No matching variant of project :react-native-async-storage_async-storage was found. The consumer was configured to find a library for use during compile-time, preferably optimized for Android, as well as attribute 'com.android.build.api.attributes.AgpVersionAttr' with value '8.6.0', attribute 'com.android.build.api.attributes.BuildTypeAttr' with value 'release', attribute 'org.jetbrains.kotlin.platform.type' with value 'androidJvm' but:
          - No variants exist.

...

* Try:
> Creating consumable variants is explained in more detail at https://docs.gradle.org/8.10.2/userguide/declaring_dependencies.html#sec:resolvable-consumable-configs.
> Review the variant matching algorithm at https://docs.gradle.org/8.10.2/userguide/variant_attributes.html#sec:abm_algorithm.
> Run with --stacktrace option to get the stack trace.
> Run with --info or --debug option to get more log output.
> Run with --scan to get full insights.
> Get more help at https://help.gradle.org.
Deprecated Gradle features were used in this build, making it incompatible with Gradle 9.0.
You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.
For more on this, please refer to https://docs.gradle.org/8.10.2/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.
18 actionable tasks: 18 executed
BUILD FAILED in 1m 27s
Error: Gradle build failed with unknown error. See logs for the "Run gradlew" phase for more information.
```

주요 원인은 다음과 같습니다.

**태스크 종속성 확인 실패**
- `:app:lintVitalReportRelease` 태스크의 종속성을 확인할 수 없음

**종속성 해결 실패**
- `:app:releaseCompileClasspath` 구성에 대한 모든 종속성을 해결하지 못함

**특정 라이브러리 문제**
- `react-native-async-storage_async-storage` 프로젝트를 해결할 수 없음.
	- 이 라이브러리는 프로젝트 `:app` 에서 필요로 함

**Variant 불일치**
- `react-native-async-storage_async-storage` 프로젝트의 variant가 프로젝트의 요구 조건과 일치하 지 않음
	- 프로젝트는 다음 조건을 만족하는 라이브러리를 찾도록 설정
		- 컴파일 타임에 사용할 라이브러리
		- 안드로이드에 최적화된 라이브러리
		- Android Gradle Plugin 버전이 8.6.0
		- `BuildTypeAttr` 값이 release
		- `org.jetbrains.kotlin.platform.type` 값이 androidJvm.


## How to solve

https://stackoverflow.com/questions/79261247/build-failed-gradle-build-failed-with-unknown-error-see-logs-for-the-run-grad

```
npx expo prebuild --clean
```

명령어를 실행하여 프로젝트의 네이티브 빌드 파일(android 및 ios)을 새로 생성하고 기존 캐시나 잘못된 설정을 정리합니다.



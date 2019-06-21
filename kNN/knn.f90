MODULE KNN
    IMPLICIT NONE
    CONTAINS
    FUNCTION EUCLIDEAN_DISTANCE(POINT_1, POINT_2) RESULT(DISTANCE)
        IMPLICIT NONE
        REAL, DIMENSION(:) :: POINT_1, POINT_2
        REAL :: DISTANCE, PARTIAL_DISTANCE = 0
        INTEGER(KIND = 16) :: QTD_ATTR, I = 1

        ! Quantidade de atributos da instância
        QTD_ATTR = SIZE(POINT_1)

     10 CONTINUE
        IF (I .LE. QTD_ATTR) THEN
            PARTIAL_DISTANCE = PARTIAL_DISTANCE + (POINT_1(I) - POINT_2(I)) ** 2
            I = I + 1
            GOTO 10
        END IF

        DISTANCE = PARTIAL_DISTANCE

    END FUNCTION
END MODULE KNN
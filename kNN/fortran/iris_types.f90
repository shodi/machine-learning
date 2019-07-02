MODULE IRIS_DATA
    IMPLICIT NONE
    TYPE :: IRIS_RECORD
        DOUBLE PRECISION :: SEPAL_LENGTH, SEPAL_WIDTH
        DOUBLE PRECISION :: PETAL_LENGTH, PETAL_WIDTH
        CHARACTER(LEN = 16) :: CLASS
    END TYPE IRIS_RECORD

    TYPE :: IRIS_RECORD_LIST
        TYPE(IRIS_RECORD) :: VALUE
        INTEGER :: LENGTH = 0
        TYPE(IRIS_RECORD_LIST), POINTER :: NEXT => NULL()
        TYPE(IRIS_RECORD_LIST), POINTER :: TAIL => NULL()
    END TYPE IRIS_RECORD_LIST

    TYPE :: TUPLE
        INTEGER :: INDEX
        REAL(KIND = 8) :: DISTANCE
        TYPE(IRIS_RECORD) :: VALUE
    END TYPE TUPLE

    CONTAINS
    FUNCTION FIND_ELEMENT(LIST, INDEX) RESULT(ELEMENT)
        IMPLICIT NONE
        TYPE(IRIS_RECORD_LIST), INTENT(IN) :: LIST
        TYPE(IRIS_RECORD_LIST), POINTER :: CUR
        INTEGER, INTENT(IN) :: INDEX
        INTEGER :: COUNTER = 1
        TYPE(IRIS_RECORD) :: ELEMENT

        CUR = LIST%NEXT
     10 CONTINUE
        IF (COUNTER .LT. INDEX) THEN
            CUR = CUR%NEXT
            COUNTER = COUNTER + 1
            GOTO 10
        END IF

        ELEMENT = CUR%VALUE

    END FUNCTION FIND_ELEMENT

    SUBROUTINE APPEND(LIST, VAL)
        IMPLICIT NONE
        TYPE(IRIS_RECORD_LIST), INTENT(INOUT) :: LIST
        TYPE(IRIS_RECORD), INTENT(IN) :: VAL

        IF (ASSOCIATED(LIST%NEXT)) THEN
            ALLOCATE(LIST%TAIL%NEXT)
            LIST%TAIL => LIST%TAIL%NEXT
        ELSE
            ALLOCATE(LIST%NEXT)
            LIST%TAIL => LIST%NEXT
        END IF

        LIST%TAIL%VALUE = VAL
        LIST%LENGTH = LIST%LENGTH + 1
    END SUBROUTINE APPEND

    SUBROUTINE PRINT_LIST(LIST)
        IMPLICIT NONE
        TYPE(IRIS_RECORD_LIST), INTENT(IN) :: LIST
        TYPE(IRIS_RECORD_LIST), POINTER :: CUR
        INTEGER :: I

        CUR => LIST%NEXT
        I = 1
        DO WHILE (ASSOCIATED(CUR))
          PRINT *, CUR%VALUE
          I = I + 1
          CUR => CUR%NEXT
        ENDDO
    END SUBROUTINE PRINT_LIST

END MODULE IRIS_DATA
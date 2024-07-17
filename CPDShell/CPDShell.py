from typing import Optional

from .Core.CPDCore import *
from .Core.scrubber.scrubber import Scrubber


class MarkedCPData:
    """Class for generating and storing marked data,
    needed in CPDShell

    :param raw_data: Data for CPD algorithms
    :type raw_data: list[float]
    :param expected_res: expected results of CPD algorithms work
    :type expected_res: danno yet, mb some sort of list or other crap
    """

    def __init__(
        self, raw_data: list[float], expected_res):  # (?) в каком виде ожидаемые резы
        """CPDShell object constructor

        :param raw_data: Data for CPD algorithms
        :type raw_data: list[float]
        :param expected_res: expected results of CPD algorithms work
        :type expected_res: danno yet, mb some sort of list or other crap
        """
        self.raw_data = raw_data
        self.expected_res = expected_res

    def generate_CP_dataset(self, distribution) -> "MarkedCPData":  # (?) в каком виде дистрибьюшн
        """Method for generating marked data, that contains CP with specific
        distribution
        :return: object with marked data
        :rtype: :class: `MarkedCPData`
        """
        return MarkedCPData([], [])


class CPDShell:
    """Class, that grants a convenient interface to
    work with CPD algorithms

    :param data: Data for CPD algorithms
    :type data: list[float], :class: `MarkedCPData`
    :param cpd_core: :class: `CPDCore` object for executing CPD algorithms
    :type cpd_core: :class: `CPDCore`
    """

    def __init__(
        self,
        data: list[float] | MarkedCPData,
        *,
        algorithm_class: Optional[type["Algorithm"]] = None,
        scrubber_class: Optional[type[Scrubber]] = None
    ) -> None:
        """CPDShell object constructor

        :param data: list of values or :class: `MarkedCPData` object for CPD
        :type data: list[float], :class: `MarkedCPData`
        """
        self.data = data
        self.cpd_core: CPDCore  # если не передали алго и скраббер, пихаем стандартные

    def change_data(self, data: list[float] | MarkedCPData) -> None:
        """Sets a new data in :class: `CPDShell` object

        :param data: list of new values or :class: `MarkedCPData` object for CPD
        :type data: list[float], :class: `MarkedCPData`
        """
        ...

    def change_scrubber(self, scrubber_class: type[Scrubber]) -> None:  # (?) в каком виде скраббер
        """Sets another scrubber in :class: `CPDShell` object"""
        ...

    def change_CPD_algorithm(self, algorithm_class: type["Algorithm"]) -> None:  # (?) в каком виде алгос
        """Sets another scrubber in :class: `CPDShell` object"""
        ...

    def change_scenario(self, n_counts, locate) -> None:  # (??) в каком виде сценарий
        """Sets another scrubber in :class: `CPDShell` object"""
        ...

    def run_CPD_algorithm(self) -> dict:  # (?) в каком виде ответ и его принт
        """Execute CPD algorithm, returns its result and prints it"""
        # Возвращает дикт с результом в зависимости от даты
        # Может сделать там доп.поле в котором будет лежать красивая строка, описывающая результат
        return dict()

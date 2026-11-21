"""
Pydantic models for Analytics-STeeR

This module defines the data models that match our ontology structure.
"""

from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum

class PeriodType(str, Enum):
    MONTH = "month"
    QUARTER = "quarter"
    SEMESTER = "semester"
    YEAR = "year"

class RoleType(str, Enum):
    LCSP = "LCSP"
    ENGAGEMENT_PARTNER = "EngagementPartner"
    ENGAGEMENT_MANAGER = "EngagementManager"
    PARTICIPATING_PARTNER = "ParticipatingPartner"
    PARTICIPATING_MANAGER = "ParticipatingManager"
    BILLING_PARTNER = "BillingPartner"
    BILLING_MANAGER = "BillingManager"
    ENGAGEMENT_CONTROLLER = "EngagementController"
    CONSULTANT = "Consultant"

class AuthorityLevel(int, Enum):
    LCSP = 1
    ENGAGEMENT_PARTNER = 2
    ENGAGEMENT_MANAGER = 3
    PARTICIPATING_PARTNER = 4
    PARTICIPATING_MANAGER = 5
    BILLING_PARTNER = 6
    BILLING_MANAGER = 7
    ENGAGEMENT_CONTROLLER = 8

class Time(BaseModel):
    """Time dimension model"""
    snapshot_period: str = Field(..., description="Period identifier (e.g., 2023-12)", csv_mapping="SNAPSHOT_PERIOD")
    year: int = Field(..., ge=2000, le=2100)
    period: int = Field(..., ge=1, le=12)
    period_type: PeriodType = Field(default=PeriodType.MONTH)
    fiscal_year: int = Field(..., ge=2000, le=2100)
    fiscal_period: int = Field(..., ge=1, le=12)
    debut_periode: date = Field(..., description="Period start date")
    fin_periode: date = Field(..., description="Period end date")
    is_current_period: bool = Field(default=False)
    fiscal_year_start_month: int = Field(default=1, ge=1, le=12)

    @validator('fiscal_period')
    def validate_fiscal_period(cls, v, values):
        if 'period_type' in values:
            if values['period_type'] == PeriodType.QUARTER and v > 4:
                raise ValueError("Fiscal period cannot exceed 4 for quarters")
            if values['period_type'] == PeriodType.SEMESTER and v > 2:
                raise ValueError("Fiscal period cannot exceed 2 for semesters")
        return v

class Client(BaseModel):
    """Client dimension model"""
    client_id: str = Field(..., min_length=1, csv_mapping="Client Id")
    client_name: str = Field(..., min_length=1, csv_mapping="Client")
    client_group_name: Optional[str] = Field(None, csv_mapping="Client Group")
    direct_parent_id: Optional[str] = Field(None, csv_mapping="Direct Parent")
    industry_sector: Optional[str] = Field(None, csv_mapping="Industrie & Secteur")
    industry: Optional[str] = Field(None, csv_mapping="Industrie")
    sector: Optional[str] = Field(None, csv_mapping="Secteur")
    region: Optional[str] = Field(None, csv_mapping="Eng. Region")
    pole: Optional[str] = Field(None, csv_mapping="Eng. Pole")

class Person(BaseModel):
    """Person dimension model"""
    person_id: str = Field(..., min_length=1)
    matricule: str = Field(..., min_length=1)
    full_name: str = Field(..., min_length=1)

class Role(BaseModel):
    """Role dimension model"""
    role_id: str = Field(..., min_length=1)
    role_type: RoleType
    authority_level: AuthorityLevel
    start_date: date
    end_date: Optional[date] = None
    person_id: str = Field(..., min_length=1)
    client_id: Optional[str] = None
    engagement_id: Optional[str] = None
    under_authority_of: Optional[str] = None

    @validator('end_date')
    def validate_dates(cls, v, values):
        if v and 'start_date' in values and v < values['start_date']:
            raise ValueError("End date must be after start date")
        return v

    @validator('authority_level')
    def validate_authority_level(cls, v, values):
        if 'role_type' in values:
            role_type = values['role_type']
            if role_type == RoleType.LCSP and v != AuthorityLevel.LCSP:
                raise ValueError("LCSP role must have LCSP authority level")
            elif role_type == RoleType.ENGAGEMENT_PARTNER and v != AuthorityLevel.ENGAGEMENT_PARTNER:
                raise ValueError("EngagementPartner role must have EngagementPartner authority level")
            elif role_type == RoleType.ENGAGEMENT_MANAGER and v != AuthorityLevel.ENGAGEMENT_MANAGER:
                raise ValueError("EngagementManager role must have EngagementManager authority level")
            elif role_type == RoleType.PARTICIPATING_PARTNER and v != AuthorityLevel.PARTICIPATING_PARTNER:
                raise ValueError("ParticipatingPartner role must have ParticipatingPartner authority level")
            elif role_type == RoleType.PARTICIPATING_MANAGER and v != AuthorityLevel.PARTICIPATING_MANAGER:
                raise ValueError("ParticipatingManager role must have ParticipatingManager authority level")
            elif role_type == RoleType.BILLING_PARTNER and v != AuthorityLevel.BILLING_PARTNER:
                raise ValueError("BillingPartner role must have BillingPartner authority level")
            elif role_type == RoleType.BILLING_MANAGER and v != AuthorityLevel.BILLING_MANAGER:
                raise ValueError("BillingManager role must have BillingManager authority level")
            elif role_type == RoleType.ENGAGEMENT_CONTROLLER and v != AuthorityLevel.ENGAGEMENT_CONTROLLER:
                raise ValueError("EngagementController role must have EngagementController authority level")
        return v

    @validator('under_authority_of')
    def validate_authority_hierarchy(cls, v, values):
        if v and 'role_type' in values and 'authority_level' in values:
            role_type = values['role_type']
            authority_level = values['authority_level']
            
            if role_type == RoleType.LCSP and v is not None:
                raise ValueError("LCSP cannot be under authority of any other role")
            elif role_type == RoleType.ENGAGEMENT_PARTNER and v is None:
                raise ValueError("EngagementPartner must be under authority of LCSP")
            elif role_type == RoleType.ENGAGEMENT_MANAGER and v is None:
                raise ValueError("EngagementManager must be under authority of EngagementPartner")
            elif role_type in [RoleType.PARTICIPATING_PARTNER, RoleType.PARTICIPATING_MANAGER,
                              RoleType.BILLING_PARTNER, RoleType.BILLING_MANAGER,
                              RoleType.ENGAGEMENT_CONTROLLER] and v is None:
                raise ValueError("Participating and other roles must be under authority of EngagementPartner")
        return v

class Engagement(BaseModel):
    """Engagement fact model"""
    engagement_id: str = Field(..., min_length=1)
    snapshot_period: str = Field(..., min_length=1)
    client_id: str = Field(..., min_length=1)
    master_engagement: Optional[str] = None
    net_revenue: float = Field(..., ge=0)
    total_costs: float = Field(..., ge=0)
    net_margin: float
    margin_percentage: float = Field(..., ge=0, le=100)

    @validator('net_margin')
    def calculate_net_margin(cls, v, values):
        if 'net_revenue' in values and 'total_costs' in values:
            return values['net_revenue'] - values['total_costs']
        return v

    @validator('margin_percentage')
    def calculate_margin_percentage(cls, v, values):
        if 'net_revenue' in values and 'net_margin' in values:
            if values['net_revenue'] > 0:
                return (values['net_margin'] / values['net_revenue']) * 100
            return 0
        return v

class PopMetrics(BaseModel):
    """Period-over-Period metrics model"""
    engagement_id: str = Field(..., min_length=1)
    snapshot_period: str = Field(..., min_length=1)
    previous_period_revenue: Optional[float] = None
    previous_year_revenue: Optional[float] = None
    pop_revenue_growth: Optional[float] = None
    yoy_revenue_growth: Optional[float] = None
    pop_margin_growth: Optional[float] = None

class SeasonalityMetrics(BaseModel):
    """Seasonality metrics model"""
    engagement_id: str = Field(..., min_length=1)
    snapshot_period: str = Field(..., min_length=1)
    seasonal_factor: Optional[float] = None
    seasonally_adjusted_revenue: Optional[float] = None
    seasonally_adjusted_growth: Optional[float] = None 